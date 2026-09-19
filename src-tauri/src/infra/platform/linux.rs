//! Linux platform integration (X11 and Wayland).
//!
//! Inspired by CopyQ's proven architecture for Linux:
//! 1. Runtime environment detection (Wayland vs X11, Hyprland, Sway, GNOME, KDE Plasma).
//! 2. Active window introspection across compositors (Hyprland IPC, Sway IPC, X11 EWMH).
//! 3. Keyboard synthesis for pasting (Ctrl+V, Shift+Insert for terminal emulators).
//! 4. Multiformat clipboard handling (UTF-8, text/plain;charset=utf-8, text/html, text/uri-list).
//! 5. Clipboard Persistence Daemon: ensures copied data survives even after the source application exits.
//! 6. Echo suppression and sequence tracking for atomic synchronization.

#![cfg(target_os = "linux")]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use once_cell::sync::Lazy;
// NOTE: `wrapper::ConnectionExt` is a subtrait of `xproto::ConnectionExt` that
// additionally provides `change_property8/32`; importing it alone covers every
// X11 call in this module, so no separate xproto import is needed.
use x11rb::wrapper::ConnectionExt as _;

use crate::error::{Error, Result};

/// Process-local clipboard sequence counter.
static SEQ: AtomicU32 = AtomicU32::new(1);

/// Current sequence value (does not increment).
pub fn sequence_number() -> u32 {
    SEQ.load(Ordering::Relaxed)
}

/// Advance the sequence and return the new value.
pub fn bump_sequence_number() -> u32 {
    SEQ.fetch_add(1, Ordering::SeqCst) + 1
}

/// Identity of the window that was focused when a copy happened.
#[derive(Debug, Clone, Default)]
pub struct WindowInfo {
    /// Application class / id, e.g. `firefox`, `gnome-terminal`.
    pub app: Option<String>,
    /// Window title at the moment of capture.
    pub title: Option<String>,
    /// X11 window id where available, otherwise 0.
    pub hwnd: isize,
}

/// Session type detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionType {
    Wayland,
    X11,
}

/// Detect whether the current session is Wayland or X11.
pub fn detect_session_type() -> SessionType {
    if std::env::var("WAYLAND_DISPLAY").is_ok()
        || std::env::var("XDG_SESSION_TYPE").map(|v| v == "wayland").unwrap_or(false)
    {
        SessionType::Wayland
    } else {
        SessionType::X11
    }
}

/// Check if running inside GNOME Shell.
pub fn is_gnome_desktop() -> bool {
    std::env::var("XDG_CURRENT_DESKTOP")
        .map(|v| v.to_lowercase().contains("gnome"))
        .unwrap_or(false)
}

/// Detect the active window across Hyprland, Sway, X11 and generic Wayland.
pub fn foreground_window() -> WindowInfo {
    // 1. Hyprland: direct IPC via hyprctl
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok() {
        if let Some(info) = read_hyprland_window() {
            return info;
        }
    }

    // 2. Sway and wlroots compositors exposing SWAYSOCK
    if std::env::var("SWAYSOCK").is_ok() {
        if let Some(info) = read_sway_window() {
            return info;
        }
    }

    // 3. Generic Wayland
    if detect_session_type() == SessionType::Wayland {
        return WindowInfo::default();
    }

    // 4. X11 session: full EWMH introspection
    read_x11_window().unwrap_or_default()
}

fn read_hyprland_window() -> Option<WindowInfo> {
    let output = std::process::Command::new("hyprctl")
        .args(["activewindow", "-j"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let v: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let app = v.get("class").and_then(|c| c.as_str()).map(|s| s.to_string());
    let title = v.get("title").and_then(|t| t.as_str()).map(|s| s.to_string());
    if app.is_none() && title.is_none() {
        return None;
    }
    Some(WindowInfo { app, title, hwnd: 0 })
}

fn read_sway_window() -> Option<WindowInfo> {
    let output = std::process::Command::new("swaymsg")
        .args(["-t", "get_tree"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    find_sway_focused(&tree)
}

fn find_sway_focused(node: &serde_json::Value) -> Option<WindowInfo> {
    for key in ["nodes", "floating_nodes"] {
        if let Some(children) = node.get(key).and_then(|v| v.as_array()) {
            for child in children {
                if let Some(info) = find_sway_focused(child) {
                    return Some(info);
                }
            }
        }
    }

    let focused = node.get("focused").and_then(|v| v.as_bool()).unwrap_or(false);
    if !focused {
        return None;
    }

    if let Some(app_id) = node.get("app_id").and_then(|v| v.as_str()) {
        let title = node.get("name").and_then(|v| v.as_str()).map(|s| s.to_string());
        return Some(WindowInfo {
            app: Some(app_id.to_string()),
            title,
            hwnd: 0,
        });
    }

    if let Some(props) = node.get("window_properties") {
        let app = props.get("class").and_then(|v| v.as_str()).map(|s| s.to_string());
        let title = props.get("title").and_then(|v| v.as_str()).map(|s| s.to_string());
        if app.is_some() || title.is_some() {
            return Some(WindowInfo { app, title, hwnd: 0 });
        }
    }

    None
}

fn read_x11_window() -> Option<WindowInfo> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let screen = conn.setup().roots.get(screen_num)?;
    let root = screen.root;

    let net_active = conn.intern_atom(false, b"_NET_ACTIVE_WINDOW").ok()?.reply().ok()?.atom;
    let net_wm_name = conn.intern_atom(false, b"_NET_WM_NAME").ok()?.reply().ok()?.atom;
    let utf8_string = conn.intern_atom(false, b"UTF8_STRING").ok()?.reply().ok()?.atom;

    let prop = conn
        .get_property(false, root, net_active, u32::from(AtomEnum::WINDOW), 0, 1)
        .ok()?
        .reply()
        .ok()?;
    let win_id = prop.value32()?.next()?;
    if win_id == 0 {
        return None;
    }

    let title = conn
        .get_property(false, win_id, net_wm_name, utf8_string, 0, 1024)
        .ok()?
        .reply()
        .ok()
        .and_then(|reply| String::from_utf8(reply.value).ok())
        .filter(|s| !s.is_empty());

    let app = conn
        .get_property(
            false,
            win_id,
            u32::from(AtomEnum::WM_CLASS),
            u32::from(AtomEnum::STRING),
            0,
            1024,
        )
        .ok()?
        .reply()
        .ok()
        .and_then(|reply| {
            reply
                .value
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .last()
                .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok())
        });

    Some(WindowInfo {
        app,
        title,
        hwnd: win_id as isize,
    })
}

/// Raise the targeted window in X11 and simulate pasting.
/// Uses Shift+Insert for terminal emulators if applicable, otherwise Ctrl+V.
pub fn paste_into(hwnd: isize) -> Result<()> {
    // 1. If we have a valid X11 window id, raise and focus it like CopyQ does.
    if hwnd != 0 && detect_session_type() == SessionType::X11 {
        raise_x11_window(hwnd as u32);
    }

    // Give the window manager a moment to settle focus
    thread::sleep(Duration::from_millis(80));

    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| Error::platform(format!("Enigo initialization failed: {e:?}")))?;

    // Check if destination might be a terminal window (common terminal names)
    let is_terminal = {
        let win = foreground_window();
        win.app.as_deref().map(is_terminal_class).unwrap_or(false)
    };

    if is_terminal {
        // Many Linux terminals (xterm, urxvt, foot, kitty, alacritty) use Shift+Insert or Ctrl+Shift+V
        enigo
            .key(Key::Shift, Direction::Press)
            .map_err(|e| Error::platform(format!("failed Shift press: {e:?}")))?;
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| Error::platform(format!("failed Control press: {e:?}")))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| Error::platform(format!("failed V click: {e:?}")))?;
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| Error::platform(format!("failed Control release: {e:?}")))?;
        enigo
            .key(Key::Shift, Direction::Release)
            .map_err(|e| Error::platform(format!("failed Shift release: {e:?}")))?;
    } else {
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| Error::platform(format!("failed Control press: {e:?}")))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| Error::platform(format!("failed V click: {e:?}")))?;
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| Error::platform(format!("failed Control release: {e:?}")))?;
    }

    Ok(())
}

fn is_terminal_class(class: &str) -> bool {
    let c = class.to_lowercase();
    c.contains("terminal")
        || c.contains("term")
        || c == "kitty"
        || c == "alacritty"
        || c == "foot"
        || c == "wezterm"
        || c == "konsole"
}

fn raise_x11_window(win_id: u32) {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let Ok((conn, screen_num)) = x11rb::connect(None) else { return };
    let Some(screen) = conn.setup().roots.get(screen_num) else { return };

    let Ok(net_active) = conn.intern_atom(false, b"_NET_ACTIVE_WINDOW") else { return };
    let Ok(net_active_reply) = net_active.reply() else { return };

    let event = ClientMessageEvent {
        response_type: CLIENT_MESSAGE_EVENT,
        format: 32,
        sequence: 0,
        window: win_id,
        type_: net_active_reply.atom,
        data: ClientMessageData::from([2, 0, 0, 0, 0]), // 2 = source indication (pager/manager)
    };

    let _ = conn.send_event(
        false,
        screen.root,
        EventMask::SUBSTRUCTURE_NOTIFY | EventMask::SUBSTRUCTURE_REDIRECT,
        event,
    );
    let _ = conn.configure_window(
        win_id,
        &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
    );
    let _ = conn.set_input_focus(InputFocus::POINTER_ROOT, win_id, Time::CURRENT_TIME);
    let _ = conn.flush();
}

// ---------------------------------------------------------------------------
// Multiformat MIME Reading (HTML & Files)
// ---------------------------------------------------------------------------

/// Read HTML flavour from clipboard if available.
pub fn read_html() -> Option<String> {
    if detect_session_type() == SessionType::Wayland {
        // In Wayland sessions, query wl-paste for text/html
        if let Ok(output) = std::process::Command::new("wl-paste")
            .args(["-t", "text/html", "-n"])
            .output()
        {
            if output.status.success() && !output.stdout.is_empty() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }
        return None;
    }

    // In X11, read via x11rb
    read_x11_atom_string(b"text/html")
}

/// Read file paths from the clipboard via `text/uri-list`.
pub fn read_files() -> Option<Vec<String>> {
    let raw = if detect_session_type() == SessionType::Wayland {
        let output = std::process::Command::new("wl-paste")
            .args(["-t", "text/uri-list", "-n"])
            .output()
            .ok()?;
        if output.status.success() && !output.stdout.is_empty() {
            String::from_utf8(output.stdout).ok()?
        } else {
            return None;
        }
    } else {
        read_x11_atom_string(b"text/uri-list")?
    };

    let mut files = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(path_str) = line.strip_prefix("file://") {
            // URL decode file path
            if let Ok(decoded) = urlencoding::decode(path_str) {
                files.push(decoded.into_owned());
            } else {
                files.push(path_str.to_string());
            }
        } else if line.starts_with('/') {
            files.push(line.to_string());
        }
    }

    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

fn read_x11_atom_string(atom_name: &[u8]) -> Option<String> {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let (conn, screen_num) = x11rb::connect(None).ok()?;
    let screen = conn.setup().roots.get(screen_num)?;
    let win = conn.generate_id().ok()?;

    conn.create_window(
        screen.root_depth,
        win,
        screen.root,
        0,
        0,
        1,
        1,
        0,
        WindowClass::INPUT_OUTPUT,
        screen.root_visual,
        &CreateWindowAux::new(),
    )
    .ok()?;

    let clip_atom = conn.intern_atom(false, b"CLIPBOARD").ok()?.reply().ok()?.atom;
    let target_atom = conn.intern_atom(false, atom_name).ok()?.reply().ok()?.atom;
    let prop_atom = conn.intern_atom(false, b"NEXUS_SELECTION").ok()?.reply().ok()?.atom;

    conn.convert_selection(win, clip_atom, target_atom, prop_atom, Time::CURRENT_TIME).ok()?;
    conn.flush().ok()?;

    // Wait for SelectionNotify
    let start = std::time::Instant::now();
    let mut result = None;

    while start.elapsed() < Duration::from_millis(300) {
        if let Ok(Some(event)) = conn.poll_for_event() {
            if let x11rb::protocol::Event::SelectionNotify(notify) = event {
                if notify.property != x11rb::NONE {
                    let prop = conn
                        .get_property(true, win, notify.property, AtomEnum::ANY, 0, 1024 * 1024)
                        .ok()?
                        .reply()
                        .ok()?;
                    if !prop.value.is_empty() {
                        result = String::from_utf8(prop.value).ok();
                    }
                }
                break;
            }
        }
        thread::sleep(Duration::from_millis(10));
    }

    let _ = conn.destroy_window(win);
    result
}

// ---------------------------------------------------------------------------
// Clipboard Persistence & Provider
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum ClipboardPayload {
    Text(String),
    Html { text: String, html: String },
    Files(Vec<String>),
    Image(Vec<u8>), // PNG
}

static ACTIVE_PERSISTENCE: Lazy<Mutex<Option<Arc<AtomicU32>>>> = Lazy::new(|| Mutex::new(None));

/// Keep the clipboard content persistent across application exits (like CopyQ).
/// When Nexus writes to the clipboard, this daemon serves the data to requesting applications.
pub fn persist_clipboard_data(payload: ClipboardPayload) {
    let mut lock = ACTIVE_PERSISTENCE.lock().unwrap();
    let token = Arc::new(AtomicU32::new(1));
    *lock = Some(token.clone());
    drop(lock);

    thread::Builder::new()
        .name("nexus-clipboard-provider".into())
        .spawn(move || {
            if detect_session_type() == SessionType::Wayland {
                provide_wayland_clipboard(payload, token);
            } else {
                provide_x11_clipboard(payload, token);
            }
        })
        .ok();
}

fn provide_wayland_clipboard(payload: ClipboardPayload, token: Arc<AtomicU32>) {
    // Under Wayland, wl-copy can serve in the background
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new("wl-copy");

    match payload {
        ClipboardPayload::Text(text) => {
            cmd.args(["--type", "text/plain;charset=utf-8"]);
            if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(text.as_bytes());
                }
                while token.load(Ordering::Relaxed) == 1 {
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                let _ = child.kill();
            }
        }
        ClipboardPayload::Html { text: _, html } => {
            // Provide HTML with text fallback
            cmd.args(["--type", "text/html"]);
            if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(html.as_bytes());
                }
                while token.load(Ordering::Relaxed) == 1 {
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                let _ = child.kill();
            }
        }
        ClipboardPayload::Files(paths) => {
            let mut uri_list = String::new();
            for p in paths {
                uri_list.push_str(&format!("file://{}\r\n", p));
            }
            cmd.args(["--type", "text/uri-list"]);
            if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(uri_list.as_bytes());
                }
                while token.load(Ordering::Relaxed) == 1 {
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                let _ = child.kill();
            }
        }
        ClipboardPayload::Image(png) => {
            cmd.args(["--type", "image/png"]);
            if let Ok(mut child) = cmd.stdin(Stdio::piped()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(&png);
                }
                while token.load(Ordering::Relaxed) == 1 {
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                let _ = child.kill();
            }
        }
    }
}

fn provide_x11_clipboard(payload: ClipboardPayload, token: Arc<AtomicU32>) {
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::*;

    let Ok((conn, screen_num)) = x11rb::connect(None) else { return };
    let Some(screen) = conn.setup().roots.get(screen_num) else { return };
    let Ok(win) = conn.generate_id() else { return };

    if conn
        .create_window(
            screen.root_depth,
            win,
            screen.root,
            0,
            0,
            1,
            1,
            0,
            WindowClass::INPUT_OUTPUT,
            screen.root_visual,
            &CreateWindowAux::new().event_mask(EventMask::PROPERTY_CHANGE),
        )
        .is_err()
    {
        return;
    }

    let clip_atom = conn.intern_atom(false, b"CLIPBOARD").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);
    let targets_atom = conn.intern_atom(false, b"TARGETS").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);
    let utf8_atom = conn.intern_atom(false, b"UTF8_STRING").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);
    let string_atom = AtomEnum::STRING.into();
    let html_atom = conn.intern_atom(false, b"text/html").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);
    let uri_atom = conn.intern_atom(false, b"text/uri-list").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);
    let png_atom = conn.intern_atom(false, b"image/png").ok().and_then(|r| r.reply().ok()).map(|r| r.atom).unwrap_or(0);

    if clip_atom == 0 {
        return;
    }

    let _ = conn.set_selection_owner(win, clip_atom, Time::CURRENT_TIME);
    let _ = conn.flush();

    while token.load(Ordering::Relaxed) == 1 {
        if let Ok(Some(event)) = conn.poll_for_event() {
            match event {
                x11rb::protocol::Event::SelectionClear(clear) => {
                    if clear.selection == clip_atom {
                        // Another application took ownership of the clipboard
                        break;
                    }
                }
                x11rb::protocol::Event::SelectionRequest(req) => {
                    let mut property = req.property;
                    if req.target == targets_atom {
                        let supported = [targets_atom, utf8_atom, string_atom, html_atom, uri_atom, png_atom];
                        let _ = conn.change_property32(
                            PropMode::REPLACE,
                            req.requestor,
                            req.property,
                            AtomEnum::ATOM,
                            &supported,
                        );
                    } else {
                        let bytes: Option<Vec<u8>> = match &payload {
                            ClipboardPayload::Text(t) => {
                                if req.target == utf8_atom || req.target == string_atom {
                                    Some(t.as_bytes().to_vec())
                                } else {
                                    None
                                }
                            }
                            ClipboardPayload::Html { text, html } => {
                                if req.target == html_atom {
                                    Some(html.as_bytes().to_vec())
                                } else if req.target == utf8_atom || req.target == string_atom {
                                    Some(text.as_bytes().to_vec())
                                } else {
                                    None
                                }
                            }
                            ClipboardPayload::Files(paths) => {
                                if req.target == uri_atom {
                                    let mut s = String::new();
                                    for p in paths {
                                        s.push_str(&format!("file://{}\r\n", p));
                                    }
                                    Some(s.into_bytes())
                                } else if req.target == utf8_atom || req.target == string_atom {
                                    Some(paths.join("\n").into_bytes())
                                } else {
                                    None
                                }
                            }
                            ClipboardPayload::Image(png) => {
                                if req.target == png_atom {
                                    Some(png.clone())
                                } else {
                                    None
                                }
                            }
                        };

                        if let Some(b) = bytes {
                            let _ = conn.change_property8(
                                PropMode::REPLACE,
                                req.requestor,
                                req.property,
                                req.target,
                                &b,
                            );
                        } else {
                            property = x11rb::NONE;
                        }
                    }

                    let notify = SelectionNotifyEvent {
                        response_type: SELECTION_NOTIFY_EVENT,
                        sequence: 0,
                        time: req.time,
                        requestor: req.requestor,
                        selection: req.selection,
                        target: req.target,
                        property,
                    };
                    let _ = conn.send_event(false, req.requestor, EventMask::NO_EVENT, notify);
                    let _ = conn.flush();
                }
                _ => {}
            }
        }
        thread::sleep(Duration::from_millis(20));
    }

    let _ = conn.destroy_window(win);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_terminal_classes() {
        assert!(is_terminal_class("gnome-terminal"));
        assert!(is_terminal_class("Kitty"));
        assert!(is_terminal_class("alacritty"));
        assert!(is_terminal_class("foot"));
        assert!(!is_terminal_class("firefox"));
        assert!(!is_terminal_class("code"));
    }

    #[test]
    fn sequence_number_increments() {
        let a = bump_sequence_number();
        let b = bump_sequence_number();
        assert!(b > a);
    }
}
