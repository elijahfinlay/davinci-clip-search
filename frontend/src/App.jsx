import { useEffect, useRef, useState } from "react";

const FALLBACK_FILTERS = ["All", "Drone", "Ground", "Interview"];
const SEARCH_SCOPE_OPTIONS = [
  { value: "all", label: "All timelines" },
  { value: "current", label: "Current timeline" },
];
const INITIAL_RESULT_BATCH = 120;
const RESULT_BATCH_SIZE = 160;

const TYPE_LABELS = {
  handheld: "ground",
};

function normalizeFilterOptions(options) {
  const seen = new Set();
  const ordered = [];
  for (const label of [...FALLBACK_FILTERS, ...(options || [])]) {
    const normalized = `${label}`.trim();
    if (!normalized) continue;
    if (seen.has(normalized.toLowerCase())) continue;
    seen.add(normalized.toLowerCase());
    ordered.push(normalized);
  }
  return ordered;
}

function formatRelativeTime(iso) {
  if (!iso) return "never";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;

  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    let message = "Request failed";
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }

  return response.json();
}

function ClipThumbnail({ type, index, thumbnail }) {
  const [imageFailed, setImageFailed] = useState(false);
  const gradients = {
    drone: ["#1a3a4a", "#2d6a7a", "#4a9ab0"],
    handheld: ["#3a2a1a", "#6a4a2a", "#a07040"],
    ground: ["#3a2a1a", "#6a4a2a", "#a07040"],
    interview: ["#2a2a3a", "#4a4a6a", "#6a6a9a"],
  };
  const colors = gradients[type] || gradients.handheld;

  useEffect(() => {
    setImageFailed(false);
  }, [thumbnail]);

  if (thumbnail && !imageFailed) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          borderRadius: 6,
          overflow: "hidden",
          background: "rgba(255,255,255,0.03)",
        }}
      >
        <img
          src={thumbnail}
          alt=""
          loading="lazy"
          decoding="async"
          onError={() => setImageFailed(true)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
            filter: "saturate(0.92) contrast(1.02)",
          }}
        />
      </div>
    );
  }

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 6,
        background: `linear-gradient(${135 + index * 20}deg, ${colors[0]}, ${colors[1]}, ${colors[2]})`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.05) 3px, rgba(0,0,0,0.05) 4px)",
        }}
      />
      <svg width="20" height="14" viewBox="0 0 20 14" fill="none">
        <rect
          x="1"
          y="1"
          width="18"
          height="12"
          rx="2"
          stroke="rgba(255,255,255,0.25)"
          strokeWidth="1.5"
          fill="none"
        />
        <polygon points="8,4 14,7 8,10" fill="rgba(255,255,255,0.25)" />
      </svg>
    </div>
  );
}

function TagPill({ label }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 4,
        fontSize: 10,
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        letterSpacing: "0.02em",
        background: "rgba(255,255,255,0.05)",
        color: "rgba(255,255,255,0.4)",
        border: "1px solid rgba(255,255,255,0.06)",
      }}
    >
      {label}
    </span>
  );
}

function CoverageChip({ label, coverage, title }) {
  if (!coverage || !coverage.total) {
    return null;
  }

  const complete = coverage.complete;
  const tone = complete
    ? {
        color: "#63c482",
        bg: "rgba(99,196,130,0.08)",
        border: "rgba(99,196,130,0.18)",
      }
    : {
        color: "#c8a257",
        bg: "rgba(200,162,87,0.08)",
        border: "rgba(200,162,87,0.18)",
      };

  return (
    <span
      title={title}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 8px",
        borderRadius: 999,
        border: `1px solid ${tone.border}`,
        background: tone.bg,
        color: tone.color,
        fontSize: 10,
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: tone.color,
          flexShrink: 0,
        }}
      />
      <span>{label}</span>
      <span>{coverage.indexed}/{coverage.total}</span>
    </span>
  );
}

function summarizeTimelineSelection(selectedTimelineOptions) {
  if (!selectedTimelineOptions.length) {
    return "Full project";
  }
  if (selectedTimelineOptions.length === 1) {
    return selectedTimelineOptions[0].timeline_name;
  }
  return `${selectedTimelineOptions.length} timelines`;
}

function TypeBadge({ type }) {
  const config = {
    drone: { color: "#4a9ab0", bg: "rgba(74,154,176,0.1)", border: "rgba(74,154,176,0.2)" },
    handheld: { color: "#b08a4a", bg: "rgba(176,138,74,0.1)", border: "rgba(176,138,74,0.2)" },
    ground: { color: "#b08a4a", bg: "rgba(176,138,74,0.1)", border: "rgba(176,138,74,0.2)" },
    interview: { color: "#8a7ab0", bg: "rgba(138,122,176,0.1)", border: "rgba(138,122,176,0.2)" },
  };
  const c = config[type] || config.handheld;
  const label = TYPE_LABELS[type] || type;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 7px",
        borderRadius: 4,
        fontSize: 10,
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        textTransform: "uppercase",
        letterSpacing: "0.06em",
        color: c.color,
        background: c.bg,
        border: `1px solid ${c.border}`,
      }}
    >
      {label}
    </span>
  );
}

function SearchIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      style={{ flexShrink: 0 }}
    >
      <circle
        cx="7"
        cy="7"
        r="5.5"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth="1.5"
      />
      <line
        x1="11"
        y1="11"
        x2="14"
        y2="14"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IndexIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" fill="none"/>
      <rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" fill="none"/>
      <rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" fill="none"/>
      <rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" strokeWidth="1.2" fill="none"/>
    </svg>
  );
}

function JumpIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
      <path d="M2 10L10 2M10 2H4M10 2V8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function ClipRow({ clip, index, isSelected, onSelect, onJump }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={() => onSelect(clip.id)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        gap: 14,
        padding: "12px 16px",
        cursor: "pointer",
        background: isSelected
          ? "rgba(255,255,255,0.04)"
          : hovered
            ? "rgba(255,255,255,0.02)"
            : "transparent",
        borderLeft: isSelected
          ? "2px solid rgba(255,255,255,0.5)"
          : "2px solid transparent",
        transition: "all 0.15s ease",
        animation: `fadeSlideIn 0.3s ease ${index * 0.05}s both`,
      }}
    >
      <div style={{ width: 80, height: 52, flexShrink: 0 }}>
        <ClipThumbnail type={clip.type} index={index} thumbnail={clip.thumbnail} />
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 4,
          }}
        >
          <span
            style={{
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              fontSize: 12,
              fontWeight: 500,
              color: "rgba(255,255,255,0.85)",
              letterSpacing: "0.01em",
            }}
          >
            {clip.filename}
          </span>
          <TypeBadge type={clip.type} />
        </div>
        <p
          style={{
            margin: 0,
            fontSize: 12,
            lineHeight: 1.5,
            color: "rgba(255,255,255,0.4)",
            fontFamily: "'DM Sans', system-ui, sans-serif",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {clip.description}
        </p>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginTop: 6,
            flexWrap: "wrap",
          }}
        >
          {clip.tags.slice(0, 4).map((tag) => (
            <TagPill key={tag} label={tag} />
          ))}
          {clip.tags.length > 4 && (
            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.25)" }}>
              +{clip.tags.length - 4}
            </span>
          )}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          justifyContent: "space-between",
          flexShrink: 0,
          minWidth: 100,
        }}
      >
        <div style={{ textAlign: "right" }}>
          <div
            style={{
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              fontSize: 11,
              color: "rgba(255,255,255,0.5)",
              letterSpacing: "0.04em",
            }}
          >
            {clip.timecode}
          </div>
          <div
            style={{
              fontFamily: "'DM Sans', system-ui, sans-serif",
              fontSize: 10,
              color: "rgba(255,255,255,0.25)",
              marginTop: 2,
            }}
          >
            {clip.timeline} · T{clip.track}
          </div>
        </div>
        <button
          onClick={(event) => {
            event.stopPropagation();
            onJump(clip);
          }}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 5,
            padding: "4px 10px",
            borderRadius: 5,
            border: "1px solid rgba(255,255,255,0.1)",
            background: isSelected || hovered
              ? "rgba(255,255,255,0.08)"
              : "transparent",
            color: "rgba(255,255,255,0.5)",
            fontSize: 10,
            fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
            cursor: "pointer",
            transition: "all 0.15s ease",
            opacity: isSelected || hovered ? 1 : 0,
            letterSpacing: "0.02em",
          }}
          onMouseEnter={(event) => {
            event.currentTarget.style.background = "rgba(255,255,255,0.12)";
            event.currentTarget.style.color = "rgba(255,255,255,0.8)";
            event.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
          }}
          onMouseLeave={(event) => {
            event.currentTarget.style.background = "rgba(255,255,255,0.08)";
            event.currentTarget.style.color = "rgba(255,255,255,0.5)";
            event.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
          }}
        >
          <JumpIcon /> Jump
        </button>
      </div>
    </div>
  );
}

export default function ResolveClipSearch() {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [activeFilter, setActiveFilter] = useState("All");
  const [searchScope, setSearchScope] = useState("all");
  const [selectedTimelineUids, setSelectedTimelineUids] = useState([]);
  const [indexScopeOpen, setIndexScopeOpen] = useState(false);
  const [results, setResults] = useState([]);
  const [renderCount, setRenderCount] = useState(INITIAL_RESULT_BATCH);
  const [status, setStatus] = useState({
    connected: false,
    message: "Connecting to Resolve...",
    current_timeline: null,
    current_timeline_uid: null,
    index: {
      total: 0,
      timelines: 0,
      last_indexed: null,
      available_types: FALLBACK_FILTERS,
      is_stale: false,
      quick_mode: false,
      project_coverage: null,
      current_timeline_coverage: null,
      timeline_options: [],
    },
    reindex: {
      running: false,
      progress: 0,
      message: null,
    },
  });
  const [toast, setToast] = useState(null);
  const inputRef = useRef(null);
  const searchAbortRef = useRef(null);
  const indexScopeRef = useRef(null);
  const loadMoreRef = useRef(null);

  const filterOptions = normalizeFilterOptions(status.index.available_types);
  const timelineOptions = status.index.timeline_options || [];
  const timelineOrder = new Map(
    timelineOptions.map((option, index) => [option.timeline_uid || option.timeline_name, index]),
  );
  const timelineOptionsByKey = new Map(
    timelineOptions.map((option) => [option.timeline_uid || option.timeline_name, option]),
  );
  const selectedTimelineOptions = selectedTimelineUids
    .map((timelineUid) => timelineOptionsByKey.get(timelineUid))
    .filter(Boolean);
  const visibleResults = results.slice(0, renderCount);
  const indexScopeLabel = summarizeTimelineSelection(selectedTimelineOptions);
  const indexScopeTitle = selectedTimelineOptions.length
    ? selectedTimelineOptions.map((option) => option.timeline_name).join(", ")
    : "Index the full project";

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!filterOptions.some((option) => option.toLowerCase() === activeFilter.toLowerCase())) {
      setActiveFilter("All");
    }
  }, [filterOptions, activeFilter]);

  useEffect(() => {
    if (searchScope === "current" && !status.current_timeline_uid) {
      setSearchScope("all");
    }
  }, [searchScope, status.current_timeline_uid]);

  useEffect(() => {
    const availableTimelineUids = new Set(
      timelineOptions.map((option) => option.timeline_uid || option.timeline_name),
    );
    setSelectedTimelineUids((current) =>
      current.filter((timelineUid) => availableTimelineUids.has(timelineUid)),
    );
  }, [timelineOptions]);

  useEffect(() => {
    if (!indexScopeOpen) return undefined;

    function handlePointerDown(event) {
      if (!indexScopeRef.current?.contains(event.target)) {
        setIndexScopeOpen(false);
      }
    }

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [indexScopeOpen]);

  useEffect(() => {
    let intervalId;
    let cancelled = false;

    async function loadStatus() {
      try {
        const nextStatus = await request("/api/status");
        if (!cancelled) {
          setStatus(nextStatus);
        }
      } catch (error) {
        if (!cancelled) {
          setStatus((current) => ({
            ...current,
            connected: false,
            message: error.message || "Unable to reach backend",
          }));
        }
      }
    }

    loadStatus();
    intervalId = window.setInterval(loadStatus, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    searchAbortRef.current?.abort?.();
    searchAbortRef.current = controller;

    if (searchScope === "current" && !status.current_timeline_uid) {
      setResults([]);
      return () => {
        controller.abort();
      };
    }

    const timerId = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams();
        params.set("q", query);
        params.set("clip_type", activeFilter);
        params.set("scope", searchScope);
        const nextResults = await request(`/api/search?${params.toString()}`, {
          signal: controller.signal,
        });
        if (!cancelled) {
          setResults(nextResults.results || []);
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        if (!cancelled) {
          setToast({
            type: "error",
            message: error.message || "Search failed",
          });
          setResults([]);
        }
      }
    }, query ? 120 : 0);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timerId);
    };
  }, [query, activeFilter, searchScope, status.current_timeline_uid]);

  useEffect(() => {
    if (!results.length) {
      setRenderCount(INITIAL_RESULT_BATCH);
      setSelectedId(null);
      return;
    }
    setRenderCount(Math.min(INITIAL_RESULT_BATCH, results.length));
    if (selectedId && results.some((clip) => clip.id === selectedId)) {
      return;
    }
    setSelectedId(results[0].id);
  }, [results, selectedId]);

  useEffect(() => {
    const selectedIndex = results.findIndex((clip) => clip.id === selectedId);
    if (selectedIndex < 0 || renderCount >= results.length) {
      return;
    }
    if (selectedIndex >= renderCount - 6) {
      setRenderCount((current) => Math.min(current + RESULT_BATCH_SIZE, results.length));
    }
  }, [renderCount, results, selectedId]);

  useEffect(() => {
    const node = loadMoreRef.current;
    if (!node || renderCount >= results.length) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (!entry?.isIntersecting) {
          return;
        }
        setRenderCount((current) => Math.min(current + RESULT_BATCH_SIZE, results.length));
      },
      {
        rootMargin: "400px 0px",
      },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [renderCount, results.length]);

  useEffect(() => {
    if (!toast) return undefined;
    const timerId = window.setTimeout(() => setToast(null), 2200);
    return () => window.clearTimeout(timerId);
  }, [toast]);

  async function handleJump(clip) {
    try {
      await request("/api/jump", {
        method: "POST",
        body: JSON.stringify({ clip_id: clip.id }),
      });
      setSelectedId(clip.id);
      setToast({
        type: "success",
        message: `Playhead jumped to ${clip.filename}`,
      });
    } catch (error) {
      setToast({
        type: "error",
        message: error.message || "Jump failed",
      });
    }
  }

  async function handleReindex() {
    try {
      const timelineUids = selectedTimelineOptions.length
        ? selectedTimelineOptions.map((option) => option.timeline_uid || option.timeline_name)
        : null;
      const timelineNames = selectedTimelineOptions.length
        ? selectedTimelineOptions.map((option) => option.timeline_name)
        : null;
      const nextState = await request("/api/reindex", {
        method: "POST",
        body: JSON.stringify({
          quick_mode: status.index.quick_mode,
          timeline_uids: timelineUids,
          timeline_names: timelineNames,
        }),
      });
      setStatus((current) => ({
        ...current,
        reindex: nextState,
      }));
      setIndexScopeOpen(false);
      setToast({
        type: "success",
        message: selectedTimelineOptions.length
          ? `Reindex started for ${selectedTimelineOptions.length === 1 ? selectedTimelineOptions[0].timeline_name : `${selectedTimelineOptions.length} timelines`}`
          : "Reindex started for full project",
      });
    } catch (error) {
      setToast({
        type: "error",
        message: error.message || "Reindex failed",
      });
    }
  }

  useEffect(() => {
    function handleKeyDown(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
        return;
      }

      if (event.key === "Escape") {
        if (indexScopeOpen) {
          event.preventDefault();
          setIndexScopeOpen(false);
          return;
        }
        if (query) {
          event.preventDefault();
          setQuery("");
        }
        return;
      }

      if (!results.length) return;

      if (event.key === "ArrowDown") {
        event.preventDefault();
        const currentIndex = results.findIndex((clip) => clip.id === selectedId);
        const nextIndex = currentIndex < 0 ? 0 : Math.min(currentIndex + 1, results.length - 1);
        setSelectedId(results[nextIndex].id);
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();
        const currentIndex = results.findIndex((clip) => clip.id === selectedId);
        const nextIndex = currentIndex <= 0 ? 0 : currentIndex - 1;
        setSelectedId(results[nextIndex].id);
      }

      if (event.key === "Enter") {
        if (event.metaKey || event.ctrlKey) return;
        event.preventDefault();
        const target = results.find((clip) => clip.id === selectedId) || results[0];
        if (target) {
          handleJump(target);
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [results, selectedId, query, indexScopeOpen]);

  function handleTimelineToggle(timelineUid) {
    setSelectedTimelineUids((current) => {
      const next = current.includes(timelineUid)
        ? current.filter((uid) => uid !== timelineUid)
        : [...current, timelineUid];
      return next.sort(
        (left, right) =>
          (timelineOrder.get(left) ?? Number.MAX_SAFE_INTEGER)
          - (timelineOrder.get(right) ?? Number.MAX_SAFE_INTEGER),
      );
    });
  }

  const statusCopy = status.reindex.running
    ? `${status.reindex.message || "indexing"} ${Math.round((status.reindex.progress || 0) * 100)}%`
    : status.index.is_stale
      ? "index stale"
      : `indexed ${formatRelativeTime(status.index.last_indexed)}`;
  const currentTimelineAvailable = Boolean(status.current_timeline_uid);

  return (
    <div
      style={{
        width: "100%",
        maxWidth: 680,
        margin: "0 auto",
        minHeight: "100vh",
        fontFamily: "'DM Sans', system-ui, sans-serif",
        color: "rgba(255,255,255,0.8)",
        background: "#0e0e10",
        position: "relative",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body, #root {
          margin: 0;
          min-height: 100%;
          background: #0e0e10;
        }

        body {
          background: #0e0e10;
        }

        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }

        @keyframes jumpFlash {
          0% { background: rgba(255,255,255,0.06); }
          50% { background: rgba(255,255,255,0.1); }
          100% { background: rgba(255,255,255,0.06); }
        }

        input::placeholder {
          color: rgba(255,255,255,0.2);
        }

        input:focus {
          outline: none;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }
      `}</style>

      <div
        style={{
          padding: "20px 20px 0",
          borderBottom: "1px solid rgba(255,255,255,0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 16,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: status.connected ? "#3fbf6f" : "#bf3f3f",
                boxShadow: status.connected
                  ? "0 0 8px rgba(63,191,111,0.4)"
                  : "0 0 8px rgba(191,63,63,0.4)",
                animation: "pulseGlow 3s ease infinite",
              }}
            />
            <span
              style={{
                fontSize: 11,
                color: "rgba(255,255,255,0.35)",
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                letterSpacing: "0.04em",
              }}
            >
              {status.connected ? "Resolve connected" : "Disconnected"}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 14,
              flexWrap: "wrap",
              justifyContent: "flex-end",
              fontSize: 11,
              color: "rgba(255,255,255,0.25)",
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
            }}
          >
            <span>{status.index.total} clips</span>
            <span>·</span>
            <span>{status.index.timelines} timelines</span>
            <span>·</span>
            <span>{statusCopy}</span>
            <CoverageChip
              label="Current"
              coverage={status.index.current_timeline_coverage}
              title={status.index.current_timeline_coverage?.label || undefined}
            />
            <CoverageChip
              label="Project"
              coverage={status.index.project_coverage}
              title={status.project_name || undefined}
            />
            <div ref={indexScopeRef} style={{ position: "relative" }}>
              <button
                onClick={() => setIndexScopeOpen((open) => !open)}
                disabled={status.reindex.running}
                title={indexScopeTitle}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "3px 8px",
                  borderRadius: 4,
                  border: "1px solid rgba(255,255,255,0.08)",
                  background: "transparent",
                  color: "rgba(255,255,255,0.3)",
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                  cursor: status.reindex.running ? "default" : "pointer",
                  transition: "all 0.15s ease",
                  opacity: status.reindex.running ? 0.55 : 1,
                  maxWidth: 190,
                }}
                onMouseEnter={(event) => {
                  if (status.reindex.running) return;
                  event.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                  event.currentTarget.style.color = "rgba(255,255,255,0.5)";
                }}
                onMouseLeave={(event) => {
                  event.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                  event.currentTarget.style.color = "rgba(255,255,255,0.3)";
                }}
              >
                <span style={{ color: "rgba(255,255,255,0.18)" }}>Scope</span>
                <span
                  style={{
                    minWidth: 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {indexScopeLabel}
                </span>
                <span style={{ fontSize: 8, color: "rgba(255,255,255,0.18)" }}>▾</span>
              </button>

              {indexScopeOpen && (
                <div
                  style={{
                    position: "absolute",
                    top: "calc(100% + 8px)",
                    right: 0,
                    width: 280,
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.08)",
                    background: "#141418",
                    boxShadow: "0 18px 40px rgba(0,0,0,0.28)",
                    padding: 6,
                    zIndex: 20,
                  }}
                >
                  <button
                    onClick={() => setSelectedTimelineUids([])}
                    style={{
                      width: "100%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 10,
                      padding: "9px 10px",
                      borderRadius: 7,
                      border: "none",
                      background:
                        selectedTimelineUids.length === 0
                          ? "rgba(255,255,255,0.07)"
                          : "transparent",
                      color:
                        selectedTimelineUids.length === 0
                          ? "rgba(255,255,255,0.8)"
                          : "rgba(255,255,255,0.55)",
                      fontSize: 11,
                      fontFamily: "'DM Sans', system-ui, sans-serif",
                      cursor: "pointer",
                      textAlign: "left",
                    }}
                  >
                    <span>Full project</span>
                    <span
                      style={{
                        color: "rgba(255,255,255,0.28)",
                        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                        fontSize: 10,
                      }}
                    >
                      {status.index.project_coverage?.total || status.index.total} clips
                    </span>
                  </button>

                  <div
                    style={{
                      height: 1,
                      background: "rgba(255,255,255,0.05)",
                      margin: "6px 4px",
                    }}
                  />

                  <div
                    style={{
                      maxHeight: 260,
                      overflowY: "auto",
                      padding: "0 2px",
                    }}
                  >
                    {timelineOptions.length === 0 ? (
                      <div
                        style={{
                          padding: "10px 8px",
                          color: "rgba(255,255,255,0.25)",
                          fontSize: 11,
                          fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                        }}
                      >
                        No timelines available
                      </div>
                    ) : (
                      timelineOptions.map((option) => {
                        const timelineKey = option.timeline_uid || option.timeline_name;
                        const selected = selectedTimelineUids.includes(timelineKey);
                        return (
                          <button
                            key={timelineKey}
                            onClick={() => handleTimelineToggle(timelineKey)}
                            style={{
                              width: "100%",
                              display: "flex",
                              alignItems: "center",
                              gap: 10,
                              padding: "9px 8px",
                              borderRadius: 7,
                              border: "none",
                              background: selected
                                ? "rgba(255,255,255,0.07)"
                                : "transparent",
                              color: selected
                                ? "rgba(255,255,255,0.82)"
                                : "rgba(255,255,255,0.52)",
                              cursor: "pointer",
                              textAlign: "left",
                            }}
                          >
                            <span
                              style={{
                                width: 13,
                                height: 13,
                                borderRadius: 3,
                                border: `1px solid ${
                                  selected
                                    ? "rgba(255,255,255,0.45)"
                                    : "rgba(255,255,255,0.15)"
                                }`,
                                background: selected
                                  ? "rgba(255,255,255,0.18)"
                                  : "transparent",
                                flexShrink: 0,
                              }}
                            />
                            <span style={{ minWidth: 0, flex: 1 }}>
                              <span
                                style={{
                                  display: "block",
                                  fontSize: 11,
                                  fontFamily: "'DM Sans', system-ui, sans-serif",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {option.timeline_name}
                              </span>
                              <span
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: 6,
                                  marginTop: 3,
                                  color: "rgba(255,255,255,0.26)",
                                  fontSize: 10,
                                  fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                                }}
                              >
                                <span>{option.indexed}/{option.total}</span>
                                {option.current && <span>current</span>}
                              </span>
                            </span>
                          </button>
                        );
                      })
                    )}
                  </div>

                  <div
                    style={{
                      padding: "8px 10px 6px",
                      color: "rgba(255,255,255,0.22)",
                      fontSize: 10,
                      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                    }}
                  >
                    {selectedTimelineUids.length
                      ? `${selectedTimelineUids.length} timeline${selectedTimelineUids.length === 1 ? "" : "s"} selected`
                      : "All timelines in project"}
                  </div>
                </div>
              )}
            </div>
            <button
              onClick={handleReindex}
              disabled={status.reindex.running}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                padding: "3px 8px",
                borderRadius: 4,
                border: "1px solid rgba(255,255,255,0.08)",
                background: "transparent",
                color: "rgba(255,255,255,0.3)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                cursor: status.reindex.running ? "default" : "pointer",
                transition: "all 0.15s ease",
                opacity: status.reindex.running ? 0.55 : 1,
              }}
              onMouseEnter={(event) => {
                if (status.reindex.running) return;
                event.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                event.currentTarget.style.color = "rgba(255,255,255,0.5)";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                event.currentTarget.style.color = "rgba(255,255,255,0.3)";
              }}
            >
              <IndexIcon /> {status.reindex.running ? "Reindexing" : "Reindex"}
            </button>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "10px 14px",
            background: "rgba(255,255,255,0.03)",
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.06)",
            marginBottom: 14,
            transition: "border-color 0.15s ease",
          }}
        >
          <SearchIcon />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search clips... try 'drone mountains' or 'close-up'"
            style={{
              flex: 1,
              border: "none",
              background: "transparent",
              color: "rgba(255,255,255,0.85)",
              fontSize: 14,
              fontFamily: "'DM Sans', system-ui, sans-serif",
              letterSpacing: "0.01em",
            }}
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              style={{
                border: "none",
                background: "rgba(255,255,255,0.06)",
                color: "rgba(255,255,255,0.3)",
                borderRadius: 4,
                width: 20,
                height: 20,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                fontSize: 12,
                lineHeight: 1,
              }}
            >
              ×
            </button>
          )}
          <span
            style={{
              fontSize: 10,
              color: "rgba(255,255,255,0.15)",
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              padding: "2px 6px",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 4,
              flexShrink: 0,
            }}
          >
            ⌘K
          </span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 2,
            paddingBottom: 14,
          }}
        >
          {filterOptions.map((filter) => (
            <button
              key={filter}
              onClick={() => setActiveFilter(filter)}
              style={{
                padding: "5px 12px",
                borderRadius: 5,
                border: "none",
                background:
                  activeFilter === filter
                    ? "rgba(255,255,255,0.08)"
                    : "transparent",
                color:
                  activeFilter === filter
                    ? "rgba(255,255,255,0.7)"
                    : "rgba(255,255,255,0.25)",
                fontSize: 11,
                fontFamily: "'DM Sans', system-ui, sans-serif",
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 0.15s ease",
                letterSpacing: "0.02em",
              }}
            >
              {filter}
            </button>
          ))}

          <div style={{ flex: 1 }} />

          <div style={{ display: "flex", gap: 2, marginRight: 12 }}>
            {SEARCH_SCOPE_OPTIONS.map((option) => {
              const disabled = option.value === "current" && !currentTimelineAvailable;
              const active = searchScope === option.value;
              return (
                <button
                  key={option.value}
                  onClick={() => {
                    if (!disabled) {
                      setSearchScope(option.value);
                    }
                  }}
                  title={
                    option.value === "current" && status.current_timeline
                      ? status.current_timeline
                      : undefined
                  }
                  style={{
                    padding: "5px 12px",
                    borderRadius: 5,
                    border: "none",
                    background: active
                      ? "rgba(255,255,255,0.08)"
                      : "transparent",
                    color: disabled
                      ? "rgba(255,255,255,0.12)"
                      : active
                        ? "rgba(255,255,255,0.7)"
                        : "rgba(255,255,255,0.25)",
                    fontSize: 11,
                    fontFamily: "'DM Sans', system-ui, sans-serif",
                    fontWeight: 500,
                    cursor: disabled ? "default" : "pointer",
                    transition: "all 0.15s ease",
                    letterSpacing: "0.02em",
                  }}
                >
                  {option.label}
                </button>
              );
            })}
          </div>

          <span
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.2)",
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              alignSelf: "center",
            }}
          >
            {results.length} result{results.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      <div>
        {results.length === 0 ? (
          <div
            style={{
              padding: "60px 20px",
              textAlign: "center",
              animation: "fadeSlideIn 0.3s ease both",
            }}
          >
            <div
              style={{
                fontSize: 13,
                color: "rgba(255,255,255,0.2)",
                fontFamily: "'DM Sans', system-ui, sans-serif",
              }}
            >
              {status.index.total === 0
                ? "No index found yet"
                : query
                  ? `No clips match "${query}"`
                  : "No clips found"}
            </div>
            <div
              style={{
                fontSize: 11,
                color: "rgba(255,255,255,0.12)",
                marginTop: 6,
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              }}
            >
              {status.index.total === 0
                ? "Run Reindex to build the local clip index"
                : "Try a different search or adjust filters"}
            </div>
          </div>
        ) : (
          visibleResults.map((clip, index) => (
            <div
              key={clip.id}
              style={{
                borderBottom: "1px solid rgba(255,255,255,0.03)",
                contentVisibility: "auto",
                containIntrinsicSize: "112px",
                animation:
                  toast?.type === "success" && selectedId === clip.id
                    ? "jumpFlash 0.6s ease"
                    : "none",
              }}
            >
              <ClipRow
                clip={clip}
                index={index}
                isSelected={selectedId === clip.id}
                onSelect={setSelectedId}
                onJump={handleJump}
              />
            </div>
          ))
        )}
      </div>

      {renderCount < results.length && (
        <div
          ref={loadMoreRef}
          style={{
            padding: "18px 20px 34px",
            textAlign: "center",
            color: "rgba(255,255,255,0.18)",
            fontSize: 10,
            fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
            letterSpacing: "0.02em",
          }}
        >
          Loading more results...
        </div>
      )}

      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            padding: "8px 16px",
            borderRadius: 8,
            background: "rgba(20,20,22,0.95)",
            border: "1px solid rgba(255,255,255,0.1)",
            backdropFilter: "blur(12px)",
            display: "flex",
            alignItems: "center",
            gap: 8,
            animation: "fadeSlideIn 0.2s ease both",
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
          }}
        >
          <div
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: toast.type === "success" ? "#3fbf6f" : "#bf3f3f",
              boxShadow: toast.type === "success"
                ? "0 0 6px rgba(63,191,111,0.4)"
                : "0 0 6px rgba(191,63,63,0.4)",
            }}
          />
          <span
            style={{
              fontSize: 12,
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              color: "rgba(255,255,255,0.6)",
              letterSpacing: "0.02em",
            }}
          >
            {toast.message}
          </span>
        </div>
      )}

      <div
        style={{
          position: "fixed",
          bottom: 24,
          right: 24,
          display: toast ? "none" : "flex",
          gap: 12,
          fontSize: 10,
          color: "rgba(255,255,255,0.12)",
          fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        }}
      >
        <span>
          <span style={{ padding: "1px 5px", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 3, marginRight: 4 }}>↑↓</span>
          navigate
        </span>
        <span>
          <span style={{ padding: "1px 5px", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 3, marginRight: 4 }}>↵</span>
          jump
        </span>
        <span>
          <span style={{ padding: "1px 5px", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 3, marginRight: 4 }}>esc</span>
          clear
        </span>
      </div>
    </div>
  );
}
