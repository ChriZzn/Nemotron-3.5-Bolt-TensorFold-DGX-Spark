//! Nemotron 3.5 Bolt on the native CUDA host. libbolt.so does the forward; this file is the lane backend over it:
//! prompt pass, verify windows, rollback, the LiLiCorr drafts it holds for the next round, and the keyed draws.
//! libbolt keeps BOLT_SEQS independent sequences; each live stream owns one, and a shared round carries every stream's
//! window in one forward.
const std = @import("std");
const lanes = @import("lanes");
const reuse = @import("cuda_reuse.zig");
const cuda = @import("cuda");

pub const model_type = "nemotron_h_omni";
pub const formats: []const []const u8 = &.{"modelopt-mixed"};
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 2048;
pub const lone_min_budget: u64 = 0;
pub const warm_quiet: u64 = 0;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

/// Rows a verify window carries at most: the pending token and seven drafts.
const window: u32 = 8;
/// Prompt rows a cancel or yield waits for at most, and the piece while other streams are decoding (their rounds run
/// between pieces).
const piece_rows: usize = 2048;
const busy_piece_rows: usize = 512;
/// Streams and rows one shared forward carries (libbolt's SEG_MAX and BATCH_MAX).
const max_segs: usize = 16;
const max_batch: u32 = 128;
/// Picture runs one prompt piece may hold, and frames one video may have.
const media_runs_max = 64;
const max_frames = 512;
/// Candidates kept per row for the keyed draw.
const cand_max: usize = 64;

/// Acceptance by depth until a stream has its own (greedy replies on the drafter's own text: 0.86 a level).
const prior = [_]f64{ 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68 };

extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(clk: c_int, ts: *Timespec) c_int;
fn nowMs() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1000.0 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// A run of prompt tokens whose rows come from a picture's embeddings (bolt_prefill_seq_media).
const MediaRun = extern struct { pos: c_int, count: c_int, embed: ?*anyopaque, row: c_int };

/// libbolt's C ABI, resolved once.
const Api = struct {
    open: *const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque,
    free: *const fn (?*anyopaque) callconv(.c) void,
    seqs: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_alloc: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_release: *const fn (?*anyopaque, c_int) callconv(.c) void,
    seq_bytes: *const fn (?*anyopaque) callconv(.c) u64,
    prompt_end: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    prefill: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int,
    forward: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int,
    forward_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int,
    candidates: *const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void,
    keep: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    propose: *const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int,
    propose_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void,
    has_draft: *const fn (?*anyopaque) callconv(.c) c_int,
    vision_ready: *const fn (?*anyopaque) callconv(.c) c_int,
    image_decode: *const fn ([*]const u8, usize, *?[*]u8, *c_int, *c_int) callconv(.c) c_int,
    free_host: *const fn (?*anyopaque) callconv(.c) void,
    image_grid: *const fn (c_int, c_int, *c_int, *c_int) callconv(.c) c_int,
    embed_new: *const fn (?*anyopaque, [*]const [*]const u8, [*]const c_int, [*]const c_int, c_int, c_int, c_int, *c_int) callconv(.c) ?*anyopaque,
    embed_free: *const fn (?*anyopaque) callconv(.c) void,
    video_decode: *const fn ([*]const u8, usize, f64, c_int, *?[*]u8, *c_int, *c_int, *c_int, *?[*]f32, *c_int, *c_int, *c_int) callconv(.c) c_int,
    prefill_media: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, [*]const MediaRun, c_int) callconv(.c) c_int,
    cost: *const fn (?*anyopaque, c_int) callconv(.c) f64,
    draft_multi_cost: *const fn (?*anyopaque) callconv(.c) f64,
    shared_costs: *const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int,
    snap_bytes: *const fn (?*anyopaque, c_int) callconv(.c) u64,
    snap_save: *const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque,
    snap_restore: *const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    snap_drop: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void,

    fn load(handle: ?*anyopaque) !Api {
        return .{
            .open = try sym(*const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque, handle, "bolt_open"),
            .free = try sym(*const fn (?*anyopaque) callconv(.c) void, handle, "bolt_free"),
            .seqs = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_seqs"),
            .seq_alloc = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_seq_alloc"),
            .seq_release = try sym(*const fn (?*anyopaque, c_int) callconv(.c) void, handle, "bolt_seq_release"),
            .seq_bytes = try sym(*const fn (?*anyopaque) callconv(.c) u64, handle, "bolt_seq_bytes"),
            .prompt_end = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "bolt_prompt_end"),
            .prefill = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int, handle, "bolt_prefill_seq"),
            .forward = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int, handle, "bolt_forward_seq"),
            .forward_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int, handle, "bolt_forward_multi"),
            .candidates = try sym(*const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void, handle, "bolt_candidates"),
            .keep = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "bolt_keep_seq"),
            .propose = try sym(*const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int, handle, "bolt_propose_seq"),
            .propose_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void, handle, "bolt_propose_multi"),
            .has_draft = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_has_draft"),
            .vision_ready = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_vision_ready"),
            .image_decode = try sym(*const fn ([*]const u8, usize, *?[*]u8, *c_int, *c_int) callconv(.c) c_int, handle, "bolt_image_decode"),
            .free_host = try sym(*const fn (?*anyopaque) callconv(.c) void, handle, "bolt_free_host"),
            .image_grid = try sym(*const fn (c_int, c_int, *c_int, *c_int) callconv(.c) c_int, handle, "bolt_image_grid"),
            .embed_new = try sym(*const fn (?*anyopaque, [*]const [*]const u8, [*]const c_int, [*]const c_int, c_int, c_int, c_int, *c_int) callconv(.c) ?*anyopaque, handle, "bolt_embed_new"),
            .embed_free = try sym(*const fn (?*anyopaque) callconv(.c) void, handle, "bolt_embed_free"),
            .video_decode = try sym(*const fn ([*]const u8, usize, f64, c_int, *?[*]u8, *c_int, *c_int, *c_int, *?[*]f32, *c_int, *c_int, *c_int) callconv(.c) c_int, handle, "bolt_video_decode"),
            .prefill_media = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, [*]const MediaRun, c_int) callconv(.c) c_int, handle, "bolt_prefill_seq_media"),
            .cost = try sym(*const fn (?*anyopaque, c_int) callconv(.c) f64, handle, "bolt_cost"),
            .draft_multi_cost = try sym(*const fn (?*anyopaque) callconv(.c) f64, handle, "bolt_draft_multi_cost"),
            .shared_costs = try sym(*const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int, handle, "bolt_shared_costs"),
            .snap_bytes = try sym(*const fn (?*anyopaque, c_int) callconv(.c) u64, handle, "bolt_snap_bytes"),
            .snap_save = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque, handle, "bolt_snap_save_seq"),
            .snap_restore = try sym(*const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int, handle, "bolt_snap_restore_seq"),
            .snap_drop = try sym(*const fn (?*anyopaque, ?*anyopaque) callconv(.c) void, handle, "bolt_snap_drop"),
        };
    }

    fn sym(comptime T: type, handle: ?*anyopaque, name: [*:0]const u8) !T {
        const p = dlsym(handle, name) orelse {
            std.log.err("bolt: libbolt.so has no {s}", .{name});
            return error.BoltMissing;
        };
        return @ptrCast(@alignCast(p));
    }
};

/// What the backend keeps for one live stream: its libbolt sequence, the prompt pass in progress, and the drafts held.
const Slot = struct {
    seq: c_int,
    ids: []c_int = &.{}, // the prompt, while its pass runs
    at: usize = 0, // prompt tokens done
    decoding: bool = false, // the prompt pass is over
    held: [window]u32 = @splat(0), // the drafts held for the stream's next window
    held_n: u32 = 0,
    media: ?*const lanes.media.MediaSet = null, // the stream's pictures
    embeds: []?*anyopaque = &.{}, // their embeddings in libbolt, encoded when the pass first needs them
};

/// Draws wait here between the forward and the round loop reading them back.
const draw_ring = 4096;

const Engine = struct {
    gpa: std.mem.Allocator,
    api: Api,
    model: *anyopaque,
    drafts: bool,
    seqs: u32,
    slots: std.AutoHashMapUnmanaged(*const lanes.Stream, *Slot) = .empty,
    ring: [draw_ring]u32 = @splat(0),
    drawn: u64 = 0,
    decoding: u32 = 0, // streams past their prompt pass
    stat_on: bool = false, // BOLT_STATS: log the round loop's time in the backend every 100 windows
    stat: struct { rounds: u32 = 0, streams: u32 = 0, rows: u32 = 0, verify: f64 = 0, draw: f64 = 0, draft: f64 = 0, keep: f64 = 0 } = .{},
    target: reuse.Target = undefined,

    /// Candidates the keyed draw needs for this stream's sampling: 1 when greedy.
    fn candCount(s: *const lanes.Stream) c_int {
        const sm = s.sampling orelse return 1;
        if (sm.temperature <= 0) return 1;
        const k: usize = if (sm.top_k == 0 or sm.top_k > cand_max) cand_max else sm.top_k;
        return if (k <= 20) 20 else @intCast(cand_max);
    }

    fn slotOf(e: *Engine, s: *const lanes.Stream) !*Slot {
        return e.slots.get(s) orelse error.BoltNoSequence;
    }

    /// The token drawn from one row of the last forward at `position`, keyed by the stream's seed.
    fn draw(e: *Engine, s: *const lanes.Stream, row: usize, position: u64, k: c_int) !u32 {
        var vals: [cand_max]f32 = undefined;
        var ids: [cand_max]c_int = undefined;
        e.api.candidates(e.model, @intCast(row), k, &vals, &ids);
        const sm = s.sampling orelse return @intCast(ids[0]);
        if (sm.temperature <= 0) return @intCast(ids[0]);
        const n: usize = @intCast(k);
        var v: [cand_max]f64 = undefined;
        var d: [cand_max]u64 = undefined;
        for (0..n) |i| {
            v[i] = vals[i];
            d[i] = @intCast(ids[i]);
        }
        return @intCast(try lanes.sampling.choose(e.gpa, v[0..n], d[0..n], position, sm));
    }

    fn remember(e: *Engine, token: u32) u64 {
        const h = e.drawn;
        e.ring[h % draw_ring] = token;
        e.drawn += 1;
        return h;
    }

    /// Take a sequence for the stream and resume it from a kept prompt state when the cache has one for its prefix.
    fn begin(e: *Engine, s: *lanes.Stream) !*Slot {
        const seq = e.api.seq_alloc(e.model);
        if (seq < 0) return error.LaneMemory; // every sequence is in use: the stream waits at the queue's head
        const slot = try e.gpa.create(Slot);
        errdefer e.gpa.destroy(slot);
        slot.* = .{ .seq = seq };
        if (s.media) |ms| {
            slot.media = ms;
            slot.embeds = try e.gpa.alloc(?*anyopaque, ms.items.len);
            @memset(slot.embeds, null);
        }
        errdefer if (slot.embeds.len > 0) e.gpa.free(slot.embeds);
        errdefer e.api.seq_release(e.model, seq);
        try e.slots.put(e.gpa, s, slot);
        errdefer _ = e.slots.remove(s);
        const prompt = s.prompt();
        s.cached = 0;
        if (s.reuse.saved) |saved| {
            if (s.reuse.at < prompt.len and e.api.snap_restore(e.model, seq, saved) == 0) {
                slot.at = s.reuse.at;
                s.cached = @intCast(slot.at);
            } else {
                s.reuse_failed = true; // the restore refused before touching the sequence: it starts from 0
            }
        }
        e.api.prompt_end(e.model, seq, @intCast(prompt.len));
        std.log.info("bolt: prefill {d} tokens ({d} from the cache), sequence {d}", .{ prompt.len, slot.at, slot.seq });
        slot.ids = try e.gpa.alloc(c_int, prompt.len);
        for (slot.ids, prompt) |*d, t| d.* = @intCast(t);
        return slot;
    }

    /// One piece of the prompt pass, cut at the cache's marks so the lane host can keep the state there. Small enough
    /// that a cancel or a background yield lands quickly, and smaller still while other streams are decoding.
    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream) !bool {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const prompt_len = s.prompt().len;
        const slot = e.slots.get(s) orelse try e.begin(s);
        if (s.isCancelled() or s.yields()) return error.Cancelled;
        if (slot.at >= prompt_len) return true;
        const rows: usize = if (e.decoding > 0) busy_piece_rows else piece_rows;
        var end = @min(prompt_len, slot.at + rows);
        for (s.reuse.marks) |m| {
            if (m > slot.at and m < end) end = m;
        }
        const last = end == prompt_len;
        var runs: [media_runs_max]MediaRun = undefined;
        var nruns: usize = 0;
        if (slot.media) |ms| {
            for (ms.items, 0..) |item, i| {
                var row: u32 = 0;
                for (item.runs) |run| {
                    const lo: usize = @max(@as(usize, run[0]), slot.at);
                    const hi: usize = @min(@as(usize, run[0]) + run[1], end);
                    if (hi > lo) {
                        if (nruns == runs.len) return error.BoltForward;
                        if (slot.embeds[i] == null) try e.encode(slot, item, i);
                        runs[nruns] = .{ .pos = @intCast(lo - slot.at), .count = @intCast(hi - lo), .embed = slot.embeds[i], .row = @intCast(@as(usize, row) + (lo - run[0])) };
                        nruns += 1;
                    }
                    row += run[1];
                }
            }
        }
        const cand: c_int = if (last) candCount(s) else 0;
        const rc = if (nruns > 0)
            e.api.prefill_media(e.model, slot.seq, slot.ids.ptr + slot.at, @intCast(end - slot.at), cand, &runs, @intCast(nruns))
        else
            e.api.prefill(e.model, slot.seq, slot.ids.ptr + slot.at, @intCast(end - slot.at), cand);
        if (rc != 0) return error.BoltForward;
        slot.at = end;
        // a picture the pass has gone beyond gives its rows back now: fifty pictures need not sit on the device together
        if (slot.media) |ms| for (ms.items, 0..) |item, i| {
            if (slot.embeds[i] == null or item.runs.len == 0) continue;
            const tail = item.runs[item.runs.len - 1];
            if (@as(usize, tail[0]) + tail[1] <= slot.at) {
                e.api.embed_free(slot.embeds[i]);
                slot.embeds[i] = null;
            }
        };
        if (s.reuse.hook) |hk| {
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(slot.at)) != null) hk.at(hk.ptr, s, @intCast(slot.at));
        }
        if (!last) return false;
        e.gpa.free(slot.ids);
        slot.ids = &.{};
        e.dropEmbeds(slot);
        slot.decoding = true;
        e.decoding += 1;
        return true;
    }

    /// Run a picture through the vision tower and projector; its rows stay on the device until the prompt pass is beyond the picture.
    fn encode(e: *Engine, slot: *Slot, item: lanes.media.Media, index: usize) !void {
        var ptrs: [max_frames][*]const u8 = undefined;
        var ws: [max_frames]c_int = undefined;
        var hs: [max_frames]c_int = undefined;
        if (item.frames.len == 0 or item.frames.len > max_frames) return error.BoltForward;
        for (item.frames, 0..) |f, k| {
            ptrs[k] = f.rgb.ptr;
            ws[k] = @intCast(f.width);
            hs[k] = @intCast(f.height);
        }
        var ntok: c_int = 0;
        const h = e.api.embed_new(e.model, &ptrs, &ws, &hs, @intCast(item.frames.len), @intCast(item.cols), @intCast(item.rows), &ntok) orelse return error.BoltVision;
        if (ntok != @as(c_int, @intCast(item.tokens))) {
            e.api.embed_free(h);
            return error.BoltVision;
        }
        slot.embeds[index] = h;
    }

    fn dropEmbeds(e: *Engine, slot: *Slot) void {
        for (slot.embeds) |h| if (h != null) e.api.embed_free(h);
        if (slot.embeds.len > 0) e.gpa.free(slot.embeds);
        slot.embeds = &.{};
        slot.media = null;
    }

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) !void {
        while (!try prefillStep(ptr, s)) {}
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const token = try e.draw(s, 0, position, candCount(s));
        return e.remember(token);
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) !u64 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const slot = try e.slotOf(s);
        const tok: u32 = switch (feed) {
            .handle => |h| e.ring[h % draw_ring],
            .value => |v| v,
        };
        const ids = [1]c_int{@intCast(tok)};
        const k = candCount(s);
        if (e.api.forward(e.model, slot.seq, &ids, 1, 0, k) != 0) return error.BoltForward;
        const next = try e.draw(s, 0, position, k);
        return e.remember(next);
    }

    fn read(ptr: *anyopaque, handle: u64) !u32 {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        return e.ring[handle % draw_ring];
    }

    /// One forward over every window (one per stream), each row drawn at its position with its stream's sampling.
    fn verify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        if (windows.len == 0 or windows.len > max_segs) return error.WindowTooWide;
        var seqs: [max_segs]c_int = undefined;
        var counts: [max_segs]c_int = undefined;
        var toks: [max_batch]c_int = undefined;
        var total: usize = 0;
        var k: c_int = 1;
        for (windows, 0..) |w, i| {
            const slot = try e.slotOf(w.stream);
            const rows = w.rows();
            if (rows > window or w.parents != null or w.masks.len != 0 or total + rows > max_batch) return error.WindowTooWide;
            toks[total] = @intCast(w.pending);
            for (0..w.held) |j| {
                toks[total + 1 + j] = @intCast(slot.held[j]);
                out[i].drafts[j] = slot.held[j];
            }
            for (w.tokens, 0..) |t, j| toks[total + 1 + w.held + j] = @intCast(t);
            seqs[i] = slot.seq;
            counts[i] = @intCast(rows);
            total += rows;
            k = @max(k, candCount(w.stream));
        }
        if (e.seqs > 1) k = cand_max; // one candidate count keeps the captured forward graphs few
        const t0 = nowMs();
        if (e.api.forward_multi(e.model, @intCast(windows.len), &seqs, &counts, &toks, k) != 0) return error.BoltForward;
        const t1 = nowMs();
        var row: usize = 0;
        for (windows, 0..) |w, i| {
            const kw = candCount(w.stream);
            for (0..w.rows()) |r| out[i].sampled[r] = try e.draw(w.stream, row + r, w.positions[r], kw);
            row += w.rows();
            (try e.slotOf(w.stream)).held_n = 0;
        }
        if (e.stat_on) {
            e.stat.rounds += 1;
            e.stat.streams += @intCast(windows.len);
            e.stat.rows += @intCast(total);
            e.stat.verify += t1 - t0;
            e.stat.draw += nowMs() - t1;
            if (e.stat.rounds == 100) {
                const n: f64 = 100;
                std.log.info("bolt stats: 100 rounds, {d:.1} streams {d:.1} rows a round; forward {d:.1} ms, draws {d:.1} ms, drafts {d:.1} ms, keeps {d:.1} ms a round", .{ @as(f64, @floatFromInt(e.stat.streams)) / n, @as(f64, @floatFromInt(e.stat.rows)) / n, e.stat.verify / n, e.stat.draw / n, e.stat.draft / n, e.stat.keep / n });
                e.stat = .{};
            }
        }
    }

    /// Roll each stream back to its window's kept rows.
    fn keep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const t0 = nowMs();
        for (windows, paths) |w, path| {
            const slot = try e.slotOf(w.stream);
            e.api.keep(e.model, slot.seq, @intCast(path.len));
        }
        e.stat.keep += nowMs() - t0;
    }

    /// Hold the drafts for each stream's next window: LiLiCorr over the kept rows and the token that follows them, for
    /// every stream of the batch in one pass over the drafter's weights.
    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const t0 = nowMs();
        defer e.stat.draft += nowMs() - t0;
        var seqs: [max_segs]c_int = undefined;
        var anchors: [max_segs]c_int = undefined;
        var who: [max_segs]*Slot = undefined;
        var depths: [max_segs]u32 = undefined;
        var n: usize = 0;
        for (requests) |r| {
            const slot = try e.slotOf(r.stream);
            slot.held_n = 0;
            // a shared round drafts before it keeps: cut the stream back to its kept rows first (a repeat is a no-op)
            if (r.rows) |rows| e.api.keep(e.model, slot.seq, @intCast(rows.len));
            if (!e.drafts or r.depth == 0 or n == max_segs) continue;
            const pending: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse continue) {
                .handle => |h| e.ring[h % draw_ring],
                .value => |v| v,
            };
            seqs[n] = slot.seq;
            anchors[n] = @intCast(pending);
            who[n] = slot;
            depths[n] = r.depth;
            n += 1;
        }
        if (n == 0) return;
        var out: [max_segs * window]c_int = @splat(0);
        var got: [max_segs]c_int = @splat(0);
        e.api.propose_multi(e.model, @intCast(n), &seqs, &anchors, &out, &got);
        for (0..n) |i| {
            const depth: usize = @min(@as(usize, depths[i]), window - 1);
            const have: usize = @intCast(@max(0, got[i]));
            const pending: u32 = @intCast(anchors[i]);
            for (0..depth) |j| who[i].held[j] = if (j < have) @intCast(out[i * window + j]) else pending;
            who[i].held_n = @intCast(depth);
        }
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        const kv = e.slots.fetchRemove(s) orelse return;
        const slot = kv.value;
        if (slot.decoding) e.decoding -|= 1;
        e.dropEmbeds(slot);
        e.api.seq_release(e.model, slot.seq);
        if (slot.ids.len > 0) e.gpa.free(slot.ids);
        e.gpa.destroy(slot);
    }

    fn backend(e: *Engine) lanes.backend.Backend {
        const vt = struct {
            var v: lanes.backend.Backend.VTable = .{
                .prefill = Engine.prefill,
                .prefill_step = Engine.prefillStep,
                .first = Engine.first,
                .queue = Engine.queue,
                .read = Engine.read,
                .verify = Engine.verify,
                .keep = Engine.keep,
                .draft = Engine.draft,
                .release = Engine.release,
            };
        };
        return .{ .ptr = e, .vtable = &vt.v };
    }
};

// -- pictures: the hooks the server uses to decode them and lay them out in the prompt ----

fn mediaImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8) anyerror!lanes.media.Media {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    var rgb: ?[*]u8 = null;
    var w: c_int = 0;
    var h: c_int = 0;
    if (e.api.image_decode(bytes.ptr, bytes.len, &rgb, &w, &h) != 0) return error.BadImage;
    defer e.api.free_host(rgb);
    const n = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 3;
    const copy = try a.dupe(u8, rgb.?[0..n]);
    var cols: c_int = 0;
    var rows: c_int = 0;
    const tokens = e.api.image_grid(w, h, &cols, &rows);
    const frames = try a.alloc(lanes.media.Picture, 1);
    frames[0] = .{ .rgb = copy, .width = @intCast(w), .height = @intCast(h) };
    return .{ .kind = .image, .frames = frames, .cols = @intCast(cols), .rows = @intCast(rows), .tokens = @intCast(tokens) };
}

fn mediaVideo(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8) anyerror!lanes.media.Media {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    var rgb: ?[*]u8 = null;
    var times: ?[*]f32 = null;
    var nframes: c_int = 0;
    var w: c_int = 0;
    var h: c_int = 0;
    var ntimes: c_int = 0;
    var cols: c_int = 0;
    var rows: c_int = 0;
    if (e.api.video_decode(bytes.ptr, bytes.len, 0, 0, &rgb, &nframes, &w, &h, &times, &ntimes, &cols, &rows) != 0) return error.BadVideo;
    defer e.api.free_host(rgb);
    defer e.api.free_host(times);
    const frame_bytes = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 3;
    const n: usize = @intCast(nframes);
    const all = try a.dupe(u8, rgb.?[0 .. frame_bytes * n]);
    const frames = try a.alloc(lanes.media.Picture, n);
    for (frames, 0..) |*f, i| f.* = .{ .rgb = all[i * frame_bytes .. (i + 1) * frame_bytes], .width = @intCast(w), .height = @intCast(h) };
    const stamps = try a.dupe(f32, times.?[0..@intCast(ntimes)]);
    const groups: u32 = @intCast(n / 2);
    const per: u32 = @intCast(@divExact(cols * rows, 4));
    return .{ .kind = .video, .frames = frames, .cols = @intCast(cols), .rows = @intCast(rows), .tokens = groups * per, .groups = groups, .times = stamps };
}

/// The chat template writes `<image N><image>` for several pictures in a message and `<image>` for one; the processor
/// then expands each `<image>` to `<img>` + the placeholders + `</img>`.
fn mediaText(ctx: *anyopaque, a: std.mem.Allocator, item: *const lanes.media.Media, index: u32, count: u32) anyerror![]const u8 {
    _ = ctx;
    var out: std.ArrayList(u8) = .empty;
    if (item.kind == .image) {
        if (count > 1) try out.print(a, "<image {d}>", .{index + 1});
        try out.appendSlice(a, "<img>");
        for (0..item.tokens) |_| try out.appendSlice(a, "<image>");
        try out.appendSlice(a, "</img>");
        return out.items;
    }
    // a video: each tubelet (two frames) is "Frame i sampled at t seconds and frame j sampled at u seconds: " + <img>...</img>,
    // the tubelets on lines of their own (processing_nemotron_h_omni.py)
    const per = item.tokens / item.groups;
    for (0..item.groups) |g| {
        if (g > 0) try out.append(a, '\n');
        for (0..2) |j| {
            const fi = g * 2 + j;
            if (fi >= item.times.len) break;
            if (j > 0) try out.appendSlice(a, " and ");
            try out.print(a, "{s} {d} sampled at {d:.2} seconds", .{ if (j == 0) "Frame" else "frame", fi + 1, item.times[fi] });
        }
        try out.appendSlice(a, ": <img>");
        for (0..per) |_| try out.appendSlice(a, "<image>");
        try out.appendSlice(a, "</img>");
    }
    return out.items;
}

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?@import("cuda_native.zig").LoneRun = null,
    target: *reuse.Target,
    learns: ?*@import("cuda_engine.zig").Engine = null,
    learn_bytes: usize = 0,
    media: ?lanes.media.Hooks = null,
};

fn shutdown(p: *anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(p));
    e.api.free(e.model);
    e.slots.deinit(e.gpa);
    e.gpa.destroy(e);
}

// -- prompt-cache copies: the Mamba state, the attention rows and the drafter's context after `at` prompt tokens ----

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    return e.api.snap_bytes(e.model, @intCast(at));
}

/// `owner` is the lane core's stream: its sequence is the state to copy.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.BoltSnapshot));
    const slot = e.slots.get(s) orelse return error.BoltSnapshot;
    return e.api.snap_save(e.model, slot.seq, @intCast(at)) orelse error.BoltSnapshot;
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.BoltSnapshot));
    const slot = e.slots.get(s) orelse return error.BoltSnapshot;
    if (e.api.snap_restore(e.model, slot.seq, saved) != 0) return error.BoltSnapshot;
    slot.held_n = 0;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    e.api.snap_drop(e.model, saved);
}

/// What the loaded libbolt measured: forward ms by window width, and shared forwards by total rows.
var costs: [window]lanes.config.Cost = undefined;
var shared: [32]lanes.config.Cost = undefined;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = io;
    _ = ctx;
    _ = kernels;
    const handle = dlopen("/opt/tensorfold/lib/libbolt.so", 2) orelse {
        std.log.err("bolt: dlopen {s}", .{if (dlerror()) |m| std.mem.span(m) else "failed"});
        return error.BoltMissing;
    };
    const api = try Api.load(handle);
    // no drafter checkpoint is read (or its memory held) when the server runs --no-drafts
    if (!o.drafts) _ = setenv("BOLT_DRAFT_OFF", "1", 1);
    const path = try gpa.allocSentinel(u8, dir.len, 0);
    defer gpa.free(path);
    @memcpy(path[0..dir.len], dir);
    const model = api.open(path, @intCast(o.context)) orelse return error.BoltLoad;
    const e = try gpa.create(Engine);
    const nseq: u32 = @intCast(@max(1, api.seqs(model)));
    e.* = .{ .gpa = gpa, .api = api, .model = model, .drafts = o.drafts and api.has_draft(model) != 0, .seqs = nseq, .stat_on = std.c.getenv("BOLT_STATS") != null };
    for (0..window) |i| costs[i] = .{ .width = @intCast(i + 1), .ms = api.cost(model, @intCast(i + 1)) };
    const propose_ms = api.cost(model, 0);
    const propose_multi_ms = api.draft_multi_cost(model);
    var rows_buf: [32]c_int = undefined;
    var ms_buf: [32]f64 = undefined;
    const n_shared: usize = @intCast(@max(0, api.shared_costs(model, &rows_buf, &ms_buf, 32)));
    for (0..n_shared) |i| shared[i] = .{ .width = @intCast(rows_buf[i]), .ms = ms_buf[i] };
    std.log.info("bolt: windows ms {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1} {d:.1}, one proposal {d:.1} ({d} streams {d:.1}), drafts {s}, {d} sequence{s}", .{ costs[0].ms, costs[1].ms, costs[2].ms, costs[3].ms, costs[4].ms, costs[5].ms, costs[6].ms, costs[7].ms, propose_ms, nseq, propose_multi_ms, if (e.drafts) "on" else "off", nseq, if (nseq == 1) "" else "s" });
    for (shared[0..n_shared]) |c| std.log.info("bolt: shared round of {d} rows {d:.1} ms", .{ c.width, c.ms });
    return .{
        .backend = e.backend(),
        .facts = .{
            .exact_width = window,
            .gpu_tokens = false,
            .mtp = e.drafts,
            .speculate = e.drafts,
            .speculate_early = false,
            .draft_prior = &prior,
            .drafts = window - 1,
            .window_costs = &costs,
            .shared_costs = shared[0..n_shared],
            .mtp_step_ms = propose_ms / 3.0,
            .draft_streams = nseq > 1,
            .hidden_rows = nseq > 1,
            .streams_exact = true,
            .batch_rows = @min(max_batch, nseq * window),
            .max_streams = nseq,
        },
        .rows = window,
        .stream_bytes = @intCast(api.seq_bytes(model)),
        .free_streams = nseq, // their state is already resident
        .ctx = e,
        .deinit = shutdown,
        .target = @ptrCast(@alignCast(e)), // the prompt cache's opaque handle: the engine itself
        .media = if (api.vision_ready(model) != 0) .{
            .ctx = e,
            .image = mediaImage,
            .video = mediaVideo,
            .text = mediaText,
            .placeholder = 18, // <image>
            .salt_first = 100, // <SPECIAL_70> ... <SPECIAL_999>, which no text produces
            .salt_count = 900,
        } else null,
    };
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BoltLoad => "Bolt could not read this checkpoint",
        error.BoltForward => "Bolt's forward failed",
        error.WindowTooWide => "Bolt verifies chains of up to eight rows a stream, sixteen streams at a time",
        else => null,
    };
}
