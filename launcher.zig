const std = @import("std");
const windows = std.os.windows;

extern "user32" fn MessageBoxW(
    hWnd: ?windows.HWND,
    lpText: [*:0]const u16,
    lpCaption: [*:0]const u16,
    uType: u32,
) callconv(.winapi) c_int;

extern "shell32" fn ShellExecuteW(
    hwnd: ?windows.HWND,
    lpOperation: ?[*:0]const u16,
    lpFile: [*:0]const u16,
    lpParameters: ?[*:0]const u16,
    lpDirectory: ?[*:0]const u16,
    nShowCmd: c_int,
) callconv(.winapi) windows.HINSTANCE;

extern "kernel32" fn OpenProcess(
    dwDesiredAccess: u32,
    bInheritHandle: windows.BOOL,
    dwProcessId: u32,
) callconv(.winapi) ?windows.HANDLE;

extern "kernel32" fn GetExitCodeProcess(
    hProcess: windows.HANDLE,
    lpExitCode: *u32,
) callconv(.winapi) windows.BOOL;

extern "kernel32" fn CloseHandle(hObject: windows.HANDLE) callconv(.winapi) windows.BOOL;

const MB_OK: u32 = 0x00000000;
const MB_ICONINFORMATION: u32 = 0x00000040;
const MB_ICONERROR: u32 = 0x00000010;
const SW_HIDE: c_int = 0;
const SW_SHOWNORMAL: c_int = 1;
const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
const STILL_ACTIVE: u32 = 259;

fn toWideZ(allocator: std.mem.Allocator, text: []const u8) ![:0]u16 {
    return std.unicode.wtf8ToWtf16LeAllocZ(allocator, text);
}

fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.accessAbsolute(io, path, .{}) catch return false;
    return true;
}

fn showMessage(allocator: std.mem.Allocator, text: []const u8, icon: u32) void {
    const w_text = toWideZ(allocator, text) catch return;
    defer allocator.free(w_text);
    const w_title = toWideZ(allocator, "Pigeon Score Scan") catch return;
    defer allocator.free(w_title);
    _ = MessageBoxW(null, w_text.ptr, w_title.ptr, MB_OK | icon);
}

fn openUrl(allocator: std.mem.Allocator, url: []const u8, root: []const u8) void {
    const w_url = toWideZ(allocator, url) catch return;
    defer allocator.free(w_url);
    const w_root = toWideZ(allocator, root) catch return;
    defer allocator.free(w_root);
    const w_open = toWideZ(allocator, "open") catch return;
    defer allocator.free(w_open);
    _ = ShellExecuteW(null, w_open.ptr, w_url.ptr, null, w_root.ptr, SW_SHOWNORMAL);
}

fn launchHidden(allocator: std.mem.Allocator, script: []const u8, root: []const u8) bool {
    const w_script = toWideZ(allocator, script) catch return false;
    defer allocator.free(w_script);
    const w_root = toWideZ(allocator, root) catch return false;
    defer allocator.free(w_root);
    const w_open = toWideZ(allocator, "open") catch return false;
    defer allocator.free(w_open);
    const result = ShellExecuteW(null, w_open.ptr, w_script.ptr, null, w_root.ptr, SW_HIDE);
    return @intFromPtr(result) > 32;
}

fn processIsAlive(pid: u32) bool {
    const handle = OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION,
        .FALSE,
        pid,
    ) orelse return false;
    defer _ = CloseHandle(handle);
    var exit_code: u32 = 0;
    if (GetExitCodeProcess(handle, &exit_code) == .FALSE) return false;
    return exit_code == STILL_ACTIVE;
}

fn openReadyUrl(
    io: std.Io,
    allocator: std.mem.Allocator,
    ready: []const u8,
    root: []const u8,
    allow_browser_fallback: bool,
) bool {
    const content = std.Io.Dir.cwd().readFileAlloc(
        io,
        ready,
        allocator,
        .limited(4096),
    ) catch return false;
    defer allocator.free(content);
    var lines = std.mem.tokenizeAny(u8, content, "\r\n");
    const url = lines.next() orelse return false;
    const pid_text = lines.next() orelse return false;
    const pid = std.fmt.parseInt(u32, std.mem.trim(u8, pid_text, " \t"), 10) catch return false;
    if (!processIsAlive(pid)) return false;
    const desktop_active = std.fs.path.join(
        allocator,
        &.{ root, "runtime", "desktop.active" },
    ) catch return false;
    defer allocator.free(desktop_active);
    const show_window = std.fs.path.join(
        allocator,
        &.{ root, "runtime", "show-window.cmd" },
    ) catch return false;
    defer allocator.free(show_window);
    if (exists(io, desktop_active) and exists(io, show_window)) {
        return launchHidden(allocator, show_window, root);
    }
    if (!allow_browser_fallback) return true;
    openUrl(allocator, std.mem.trim(u8, url, " \t"), root);
    return true;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const root = try std.process.executableDirPathAlloc(init.io, allocator);
    defer allocator.free(root);
    const ready = try std.fs.path.join(allocator, &.{ root, "runtime", "ready.txt" });
    defer allocator.free(ready);
    const failed = try std.fs.path.join(allocator, &.{ root, "runtime", "start.failed" });
    defer allocator.free(failed);

    if (exists(init.io, ready)) {
        if (openReadyUrl(init.io, allocator, ready, root, true)) return;
        std.Io.Dir.deleteFileAbsolute(init.io, ready) catch {};
    }

    const runtime_python = try std.fs.path.join(
        allocator,
        &.{ root, "runtime", "python", "python.exe" },
    );
    defer allocator.free(runtime_python);
    if (!exists(init.io, runtime_python)) {
        showMessage(
            allocator,
            "The bundled Pigeon Score Scan runtime is missing.\n\nExtract the complete archive again before starting the application.",
            MB_ICONERROR,
        );
        return;
    }

    const start_cmd = try std.fs.path.join(allocator, &.{ root, "runtime", "start.cmd" });
    defer allocator.free(start_cmd);
    std.Io.Dir.deleteFileAbsolute(init.io, failed) catch {};
    if (!launchHidden(allocator, start_cmd, root)) {
        showMessage(
            allocator,
            "Pigeon Score Scan could not start runtime\\start.cmd.",
            MB_ICONERROR,
        );
        return;
    }

    var attempts: usize = 0;
    while (attempts < 3600) : (attempts += 1) {
        if (exists(init.io, ready)) {
            // The service owns first-launch presentation: it either creates the
            // native window or opens the browser after an explicit desktop-shell
            // failure.  Opening the URL here races desktop.active and can create
            // an unwanted duplicate browser window.
            if (openReadyUrl(init.io, allocator, ready, root, false)) return;
            std.Io.Dir.deleteFileAbsolute(init.io, ready) catch {};
        }
        if (exists(init.io, failed)) {
            showMessage(
                allocator,
                "Pigeon Score Scan could not start. See runtime\\launcher.log for details.",
                MB_ICONERROR,
            );
            return;
        }
        init.io.sleep(.fromMilliseconds(500), .awake) catch {};
    }

    showMessage(
        allocator,
        "Pigeon Score Scan timed out while starting. Check runtime\\launcher.log.",
        MB_ICONERROR,
    );
}
