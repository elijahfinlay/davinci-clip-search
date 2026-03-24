import { useEffect, useRef, useState } from "react";

const FALLBACK_FILTERS = ["All", "Drone", "Ground", "Interview"];
const SEARCH_SCOPE_OPTIONS = [
  { value: "all", label: "All timelines" },
  { value: "current", label: "Current timeline" },
  { value: "saved", label: "Saved timeline" },
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
  const [shouldLoad, setShouldLoad] = useState(false);
  const containerRef = useRef(null);
  const gradients = {
    drone: ["#1a3a4a", "#2d6a7a", "#4a9ab0"],
    handheld: ["#3a2a1a", "#6a4a2a", "#a07040"],
    ground: ["#3a2a1a", "#6a4a2a", "#a07040"],
    interview: ["#2a2a3a", "#4a4a6a", "#6a6a9a"],
  };
  const colors = gradients[type] || gradients.handheld;

  useEffect(() => {
    setImageFailed(false);
    setShouldLoad(false);
  }, [thumbnail]);

  useEffect(() => {
    if (!thumbnail || shouldLoad) {
      return undefined;
    }

    const node = containerRef.current;
    if (!node || typeof window === "undefined" || !("IntersectionObserver" in window)) {
      setShouldLoad(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (!entry?.isIntersecting) {
          return;
        }
        setShouldLoad(true);
        observer.disconnect();
      },
      {
        rootMargin: "240px 0px",
      },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [shouldLoad, thumbnail]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 6,
        position: "relative",
        overflow: "hidden",
        background: thumbnail && shouldLoad && !imageFailed
          ? "rgba(255,255,255,0.03)"
          : `linear-gradient(${135 + index * 20}deg, ${colors[0]}, ${colors[1]}, ${colors[2]})`,
      }}
    >
      {thumbnail && shouldLoad && !imageFailed ? (
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
      ) : (
        <>
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.05) 3px, rgba(0,0,0,0.05) 4px)",
            }}
          />
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
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
        </>
      )}
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

function mergeLiveClipItems(currentItems, nextItem, limit = 18) {
  const merged = [nextItem, ...currentItems.filter((item) => item.id !== nextItem.id)];
  return merged.slice(0, limit);
}

function mergeResultsWithLiveItems(results, liveItems) {
  if (!liveItems.length) {
    return results;
  }
  const liveIds = new Set(liveItems.map((item) => item.id));
  return [...liveItems, ...results.filter((item) => !liveIds.has(item.id))];
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
      onClick={() => {
        onSelect(clip.id);
        onJump(clip);
      }}
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
  const [selectedSearchTimelineUid, setSelectedSearchTimelineUid] = useState(null);
  const [searchTimelineScopeOpen, setSearchTimelineScopeOpen] = useState(false);
  const [searchTimelineQuery, setSearchTimelineQuery] = useState("");
  const [selectedTimelineUids, setSelectedTimelineUids] = useState([]);
  const [indexScopeOpen, setIndexScopeOpen] = useState(false);
  const [timelineSearchQuery, setTimelineSearchQuery] = useState("");
  const [results, setResults] = useState([]);
  const [liveClips, setLiveClips] = useState([]);
  const [liveSearchTick, setLiveSearchTick] = useState(0);
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
      enrichment_running: false,
      progress: 0,
      enrichment_progress: 0,
      message: null,
      finished_at: null,
      enriched_clips: 0,
      total_enrichment_clips: 0,
      active_clip_index: 0,
      active_clip_name: null,
      latest_clip: null,
      latest_clip_stage: null,
    },
  });
  const [toast, setToast] = useState(null);
  const inputRef = useRef(null);
  const searchAbortRef = useRef(null);
  const searchTimelineScopeRef = useRef(null);
  const searchTimelineInputRef = useRef(null);
  const indexScopeRef = useRef(null);
  const timelineSearchInputRef = useRef(null);
  const loadMoreRef = useRef(null);
  const hasInitializedSearchTimelineRef = useRef(false);
  const hasInitializedTimelineSelectionRef = useRef(false);

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
  const savedSearchTimelineOption = selectedSearchTimelineUid
    ? (timelineOptionsByKey.get(selectedSearchTimelineUid) || null)
    : null;
  const normalizedSearchTimelineQuery = searchTimelineQuery.trim().toLowerCase();
  const filteredSearchTimelineOptions = normalizedSearchTimelineQuery
    ? timelineOptions.filter((option) =>
        option.timeline_name.toLowerCase().includes(normalizedSearchTimelineQuery),
      )
    : timelineOptions;
  const normalizedTimelineSearchQuery = timelineSearchQuery.trim().toLowerCase();
  const filteredTimelineOptions = normalizedTimelineSearchQuery
    ? timelineOptions.filter((option) =>
        option.timeline_name.toLowerCase().includes(normalizedTimelineSearchQuery),
      )
    : timelineOptions;
  const indexScopeLabel = summarizeTimelineSelection(selectedTimelineOptions);
  const selectedScopeClipTotal = selectedTimelineOptions.length
    ? selectedTimelineOptions.reduce((sum, option) => sum + (option.total || 0), 0)
    : (status.index.project_coverage?.total || status.index.total || 0);
  const indexScopeCountLabel = `${selectedScopeClipTotal.toLocaleString()} clips`;
  const indexScopeTitle = selectedTimelineOptions.length
    ? `${selectedTimelineOptions.map((option) => option.timeline_name).join(", ")} • ${indexScopeCountLabel}`
    : `Index the full project • ${indexScopeCountLabel}`;
  const isBusy = status.reindex.running || status.reindex.enrichment_running;
  const isEnriching = status.reindex.enrichment_running && !status.reindex.running;
  const isStopping = isBusy && `${status.reindex.message || ""}`.toLowerCase().startsWith("stopping");
  const savedSearchTimelineLabel = savedSearchTimelineOption
    ? savedSearchTimelineOption.timeline_name
    : "Select timeline";
  const savedSearchTimelineCountLabel = savedSearchTimelineOption
    ? `${(savedSearchTimelineOption.total || 0).toLocaleString()} clips`
    : "No timeline";
  const shouldStreamIntoMainResults = !query.trim() && activeFilter === "All" && searchScope === "all";
  const displayedResults = shouldStreamIntoMainResults && isBusy
    ? mergeResultsWithLiveItems(results, liveClips)
    : results;
  const visibleResults = displayedResults.slice(0, renderCount);
  const canLiveRefreshSearch = Boolean(
    query.trim()
    || searchScope === "current"
    || searchScope === "saved"
    || activeFilter !== "All"
    || (results.length > 0 && results.length <= 250),
  );

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
    setSelectedSearchTimelineUid((current) => (
      current && availableTimelineUids.has(current)
        ? current
        : null
    ));
  }, [timelineOptions]);

  useEffect(() => {
    if (hasInitializedTimelineSelectionRef.current) {
      return;
    }
    if (!status.current_timeline_uid || !timelineOptions.length) {
      return;
    }

    const currentTimelineOption = timelineOptions.find(
      (option) =>
        (option.timeline_uid || option.timeline_name) === status.current_timeline_uid
        || option.current,
    );
    if (!currentTimelineOption) {
      return;
    }

    hasInitializedTimelineSelectionRef.current = true;
    setSelectedTimelineUids([
      currentTimelineOption.timeline_uid || currentTimelineOption.timeline_name,
    ]);
  }, [status.current_timeline_uid, timelineOptions]);

  useEffect(() => {
    if (hasInitializedSearchTimelineRef.current) {
      return;
    }
    if (!timelineOptions.length) {
      return;
    }

    const currentTimelineOption = timelineOptions.find(
      (option) =>
        (option.timeline_uid || option.timeline_name) === status.current_timeline_uid
        || option.current,
    );
    const defaultOption = currentTimelineOption || timelineOptions[0];
    if (!defaultOption) {
      return;
    }

    hasInitializedSearchTimelineRef.current = true;
    setSelectedSearchTimelineUid(defaultOption.timeline_uid || defaultOption.timeline_name);
  }, [status.current_timeline_uid, timelineOptions]);

  useEffect(() => {
    if (selectedSearchTimelineUid || !timelineOptions.length) {
      return;
    }

    const currentTimelineOption = timelineOptions.find(
      (option) =>
        (option.timeline_uid || option.timeline_name) === status.current_timeline_uid
        || option.current,
    );
    const defaultOption = currentTimelineOption || timelineOptions[0];
    if (!defaultOption) {
      return;
    }

    setSelectedSearchTimelineUid(defaultOption.timeline_uid || defaultOption.timeline_name);
  }, [selectedSearchTimelineUid, status.current_timeline_uid, timelineOptions]);

  useEffect(() => {
    if (searchScope !== "saved") {
      setSearchTimelineScopeOpen(false);
    }
  }, [searchScope]);

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
    if (!searchTimelineScopeOpen) return undefined;

    function handlePointerDown(event) {
      if (!searchTimelineScopeRef.current?.contains(event.target)) {
        setSearchTimelineScopeOpen(false);
      }
    }

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [searchTimelineScopeOpen]);

  useEffect(() => {
    if (!indexScopeOpen) {
      setTimelineSearchQuery("");
      return;
    }

    const timeoutId = window.setTimeout(() => {
      timelineSearchInputRef.current?.focus();
      timelineSearchInputRef.current?.select();
    }, 0);

    return () => window.clearTimeout(timeoutId);
  }, [indexScopeOpen]);

  useEffect(() => {
    if (!searchTimelineScopeOpen) {
      setSearchTimelineQuery("");
      return;
    }

    const timeoutId = window.setTimeout(() => {
      searchTimelineInputRef.current?.focus();
      searchTimelineInputRef.current?.select();
    }, 0);

    return () => window.clearTimeout(timeoutId);
  }, [searchTimelineScopeOpen]);

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
    const stream = new EventSource("/api/reindex/stream");

    stream.onmessage = (event) => {
      if (!event.data) return;
      try {
        const nextState = JSON.parse(event.data);
        setStatus((current) => ({
          ...current,
          reindex: {
            ...current.reindex,
            ...nextState,
          },
        }));
      } catch {
        // Keep the polling fallback if the stream sends malformed data.
      }
    };

    return () => stream.close();
  }, []);

  useEffect(() => {
    if (!isBusy) {
      setLiveClips([]);
      return;
    }
    if (!status.reindex.latest_clip) {
      return;
    }
    const nextLiveClip = {
      ...status.reindex.latest_clip,
      liveStage: status.reindex.latest_clip_stage,
    };
    setLiveClips((current) => mergeLiveClipItems(current, nextLiveClip));
  }, [
    status.reindex.latest_clip,
    status.reindex.latest_clip_stage,
    isBusy,
    status.reindex.started_at,
  ]);

  useEffect(() => {
    if (!isBusy) {
      if (status.reindex.finished_at) {
        setLiveSearchTick((current) => current + 1);
      }
      return undefined;
    }
    if (!canLiveRefreshSearch) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setLiveSearchTick((current) => current + 1);
    }, 750);

    return () => window.clearTimeout(timerId);
  }, [
    canLiveRefreshSearch,
    isBusy,
    status.reindex.enriched_clips,
    status.reindex.enrichment_progress,
    status.reindex.enrichment_running,
    status.reindex.finished_at,
    status.reindex.message,
    status.reindex.processed_clips,
    status.reindex.running,
  ]);

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
    if (searchScope === "saved" && !savedSearchTimelineOption) {
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
        if (searchScope === "saved" && savedSearchTimelineOption) {
          if (savedSearchTimelineOption.timeline_uid) {
            params.set("timeline_uid", savedSearchTimelineOption.timeline_uid);
          }
          params.set("timeline_name", savedSearchTimelineOption.timeline_name);
        }
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
  }, [query, activeFilter, searchScope, savedSearchTimelineOption, status.current_timeline_uid, liveSearchTick]);

  useEffect(() => {
    if (!displayedResults.length) {
      setRenderCount(INITIAL_RESULT_BATCH);
      setSelectedId(null);
      return;
    }
    setRenderCount(Math.min(INITIAL_RESULT_BATCH, displayedResults.length));
    if (selectedId && displayedResults.some((clip) => clip.id === selectedId)) {
      return;
    }
    setSelectedId(displayedResults[0].id);
  }, [displayedResults, selectedId]);

  useEffect(() => {
    const selectedIndex = displayedResults.findIndex((clip) => clip.id === selectedId);
    if (selectedIndex < 0 || renderCount >= displayedResults.length) {
      return;
    }
    if (selectedIndex >= renderCount - 6) {
      setRenderCount((current) => Math.min(current + RESULT_BATCH_SIZE, displayedResults.length));
    }
  }, [displayedResults, renderCount, selectedId]);

  useEffect(() => {
    const node = loadMoreRef.current;
    if (!node || renderCount >= displayedResults.length) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (!entry?.isIntersecting) {
          return;
        }
        setRenderCount((current) => Math.min(current + RESULT_BATCH_SIZE, displayedResults.length));
      },
      {
        rootMargin: "400px 0px",
      },
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, [displayedResults.length, renderCount]);

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

  async function handleCancelReindex() {
    try {
      const nextState = await request("/api/reindex/cancel", {
        method: "POST",
      });
      setStatus((current) => ({
        ...current,
        reindex: nextState,
      }));
      setToast({
        type: "success",
        message: "Stopping current job...",
      });
    } catch (error) {
      setToast({
        type: "error",
        message: error.message || "Unable to stop current job",
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
        if (searchTimelineScopeOpen) {
          event.preventDefault();
          setSearchTimelineScopeOpen(false);
          return;
        }
        if (query) {
          event.preventDefault();
          setQuery("");
        }
        return;
      }

      if (
        indexScopeRef.current?.contains(document.activeElement)
        || searchTimelineScopeRef.current?.contains(document.activeElement)
      ) {
        return;
      }

      if (!displayedResults.length) return;

      if (event.key === "ArrowDown") {
        event.preventDefault();
        const currentIndex = displayedResults.findIndex((clip) => clip.id === selectedId);
        const nextIndex = currentIndex < 0 ? 0 : Math.min(currentIndex + 1, displayedResults.length - 1);
        setSelectedId(displayedResults[nextIndex].id);
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();
        const currentIndex = displayedResults.findIndex((clip) => clip.id === selectedId);
        const nextIndex = currentIndex <= 0 ? 0 : currentIndex - 1;
        setSelectedId(displayedResults[nextIndex].id);
      }

      if (event.key === "Enter") {
        if (event.metaKey || event.ctrlKey) return;
        event.preventDefault();
        const target = displayedResults.find((clip) => clip.id === selectedId) || displayedResults[0];
        if (target) {
          handleJump(target);
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [displayedResults, selectedId, query, indexScopeOpen, searchTimelineScopeOpen]);

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

  function handleSavedSearchTimelineSelect(timelineUid) {
    setSelectedSearchTimelineUid(timelineUid);
    setSearchTimelineScopeOpen(false);
  }

  const statusCopy = status.reindex.running
    ? `${status.reindex.message || "indexing"} ${Math.round((status.reindex.progress || 0) * 100)}%`
    : status.reindex.enrichment_running
      ? `${status.reindex.message || "enriching"} ${Math.round((status.reindex.enrichment_progress || 0) * 100)}%`
    : status.index.is_stale
      ? "index stale"
      : `indexed ${formatRelativeTime(status.index.last_indexed)}`;
  const currentTimelineAvailable = Boolean(status.current_timeline_uid);
  const reindexProgress = Math.max(
    0,
    Math.min(
      status.reindex.running
        ? (status.reindex.progress || 0)
        : (status.reindex.enrichment_progress || 0),
      1,
    ),
  );
  const reindexPercent = Math.round(reindexProgress * 100);
  const reindexFillPercent = isBusy
    ? Math.max(reindexPercent, status.reindex.active_clip_index ? 1 : 0)
    : reindexPercent;
  const reindexDetail = status.reindex.running
    ? (
        status.reindex.active_clip_name
          ? `${status.reindex.processed_clips}/${status.reindex.total_clips} complete · clip ${status.reindex.active_clip_index} in progress`
          : status.reindex.total_clips
            ? `${status.reindex.processed_clips}/${status.reindex.total_clips} complete`
            : "Preparing timeline scan"
      )
    : (
        status.reindex.active_clip_name
          ? `${status.reindex.enriched_clips}/${status.reindex.total_enrichment_clips} enriched · clip ${status.reindex.active_clip_index} in progress`
          : status.reindex.total_enrichment_clips
            ? `${status.reindex.enriched_clips}/${status.reindex.total_enrichment_clips} enriched`
            : "Preparing Gemini enrichment"
      );

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
                disabled={isBusy}
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
                  cursor: isBusy ? "default" : "pointer",
                  transition: "all 0.15s ease",
                  opacity: isBusy ? 0.55 : 1,
                  maxWidth: 255,
                }}
                onMouseEnter={(event) => {
                  if (isBusy) return;
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
                    flex: 1,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {indexScopeLabel}
                </span>
                <span
                  style={{
                    color: "rgba(255,255,255,0.22)",
                    fontSize: 9,
                    flexShrink: 0,
                    whiteSpace: "nowrap",
                  }}
                >
                  {indexScopeCountLabel}
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

                  <div style={{ padding: "0 4px 8px" }}>
                    <input
                      ref={timelineSearchInputRef}
                      value={timelineSearchQuery}
                      onChange={(event) => setTimelineSearchQuery(event.target.value)}
                      placeholder="Search timelines..."
                      style={{
                        width: "100%",
                        height: 32,
                        borderRadius: 7,
                        border: "1px solid rgba(255,255,255,0.07)",
                        background: "rgba(255,255,255,0.03)",
                        color: "rgba(255,255,255,0.78)",
                        fontSize: 11,
                        fontFamily: "'DM Sans', system-ui, sans-serif",
                        padding: "0 10px",
                        boxSizing: "border-box",
                      }}
                    />
                  </div>

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
                    ) : filteredTimelineOptions.length === 0 ? (
                      <div
                        style={{
                          padding: "10px 8px",
                          color: "rgba(255,255,255,0.25)",
                          fontSize: 11,
                          fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                        }}
                      >
                        No matching timelines
                      </div>
                    ) : (
                      filteredTimelineOptions.map((option) => {
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
              disabled={isBusy}
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
                cursor: isBusy ? "default" : "pointer",
                transition: "all 0.15s ease",
                opacity: isBusy ? 0.55 : 1,
              }}
              onMouseEnter={(event) => {
                if (isBusy) return;
                event.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                event.currentTarget.style.color = "rgba(255,255,255,0.5)";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                event.currentTarget.style.color = "rgba(255,255,255,0.3)";
              }}
            >
              <IndexIcon /> {status.reindex.running ? "Reindexing" : status.reindex.enrichment_running ? "Enriching" : "Reindex"}
            </button>
            {isBusy && (
              <button
                onClick={handleCancelReindex}
                disabled={isStopping}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  padding: "3px 8px",
                  borderRadius: 4,
                  border: "1px solid rgba(196,99,99,0.18)",
                  background: "transparent",
                  color: "rgba(216,124,124,0.72)",
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                  cursor: isStopping ? "default" : "pointer",
                  transition: "all 0.15s ease",
                  opacity: isStopping ? 0.6 : 1,
                }}
                onMouseEnter={(event) => {
                  if (isStopping) return;
                  event.currentTarget.style.borderColor = "rgba(216,124,124,0.34)";
                  event.currentTarget.style.color = "rgba(236,148,148,0.9)";
                }}
                onMouseLeave={(event) => {
                  event.currentTarget.style.borderColor = "rgba(196,99,99,0.18)";
                  event.currentTarget.style.color = "rgba(216,124,124,0.72)";
                }}
              >
                {isStopping ? "Stopping" : "Stop"}
              </button>
            )}
          </div>
        </div>

        {isBusy && (
          <div
            style={{
              marginBottom: 14,
              padding: "8px 10px 9px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.06)",
              background: "rgba(255,255,255,0.02)",
              animation: "fadeSlideIn 0.24s ease both",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                marginBottom: 6,
                fontSize: 10,
                color: "rgba(255,255,255,0.42)",
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                letterSpacing: "0.03em",
              }}
            >
              <span
                style={{
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {status.reindex.message || "Indexing"}
              </span>
              <span>{reindexFillPercent}%</span>
            </div>
            <div
              style={{
                height: 4,
                borderRadius: 999,
                background: "rgba(255,255,255,0.06)",
                overflow: "hidden",
                boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.02)",
              }}
            >
              <div
                style={{
                  width: `${reindexFillPercent}%`,
                  minWidth: isBusy ? 6 : 0,
                  height: "100%",
                  borderRadius: 999,
                  background: isEnriching
                    ? "linear-gradient(90deg, rgba(92,154,214,0.45), rgba(92,154,214,0.92))"
                    : "linear-gradient(90deg, rgba(99,196,130,0.45), rgba(99,196,130,0.92))",
                  boxShadow: isEnriching
                    ? "0 0 16px rgba(92,154,214,0.2)"
                    : "0 0 16px rgba(99,196,130,0.2)",
                  transition: "width 0.18s linear",
                }}
              />
            </div>
            <div
              style={{
                marginTop: 7,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                fontSize: 10,
                color: "rgba(255,255,255,0.22)",
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                letterSpacing: "0.02em",
              }}
            >
              <span
                style={{
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {reindexDetail}
              </span>
              {status.reindex.current_timeline && <span>{status.reindex.current_timeline}</span>}
            </div>
          </div>
        )}

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
                      if (option.value === "saved") {
                        setSearchTimelineScopeOpen(true);
                      } else {
                        setSearchTimelineScopeOpen(false);
                      }
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
            {searchScope === "saved" && (
              <div ref={searchTimelineScopeRef} style={{ position: "relative", marginLeft: 4 }}>
                <button
                  onClick={() => setSearchTimelineScopeOpen((open) => !open)}
                  title={savedSearchTimelineOption?.timeline_name || undefined}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "5px 10px",
                    borderRadius: 5,
                    border: "none",
                    background: "rgba(255,255,255,0.08)",
                    color: "rgba(255,255,255,0.66)",
                    fontSize: 11,
                    fontFamily: "'DM Sans', system-ui, sans-serif",
                    fontWeight: 500,
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                    letterSpacing: "0.02em",
                    maxWidth: 250,
                  }}
                >
                  <span
                    style={{
                      minWidth: 0,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {savedSearchTimelineLabel}
                  </span>
                  <span
                    style={{
                      color: "rgba(255,255,255,0.22)",
                      fontSize: 9,
                      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                      whiteSpace: "nowrap",
                      flexShrink: 0,
                    }}
                  >
                    {savedSearchTimelineCountLabel}
                  </span>
                  <span style={{ fontSize: 8, color: "rgba(255,255,255,0.22)" }}>▾</span>
                </button>

                {searchTimelineScopeOpen && (
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
                    <div style={{ padding: "0 4px 8px" }}>
                      <input
                        ref={searchTimelineInputRef}
                        value={searchTimelineQuery}
                        onChange={(event) => setSearchTimelineQuery(event.target.value)}
                        placeholder="Search saved timelines..."
                        style={{
                          width: "100%",
                          height: 32,
                          borderRadius: 7,
                          border: "1px solid rgba(255,255,255,0.07)",
                          background: "rgba(255,255,255,0.03)",
                          color: "rgba(255,255,255,0.78)",
                          fontSize: 11,
                          fontFamily: "'DM Sans', system-ui, sans-serif",
                          padding: "0 10px",
                          boxSizing: "border-box",
                        }}
                      />
                    </div>

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
                          No saved timelines available
                        </div>
                      ) : filteredSearchTimelineOptions.length === 0 ? (
                        <div
                          style={{
                            padding: "10px 8px",
                            color: "rgba(255,255,255,0.25)",
                            fontSize: 11,
                            fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                          }}
                        >
                          No matching saved timelines
                        </div>
                      ) : (
                        filteredSearchTimelineOptions.map((option) => {
                          const timelineKey = option.timeline_uid || option.timeline_name;
                          const selected = selectedSearchTimelineUid === timelineKey;
                          return (
                            <button
                              key={`saved-search-${timelineKey}`}
                              onClick={() => handleSavedSearchTimelineSelect(timelineKey)}
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
                                  {option.current && <span>current in Resolve</span>}
                                </span>
                              </span>
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <span
            style={{
              fontSize: 11,
              color: "rgba(255,255,255,0.2)",
              fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
              alignSelf: "center",
            }}
          >
            {displayedResults.length} result{displayedResults.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      <div>
        {isBusy && liveClips.length > 0 && !shouldStreamIntoMainResults && (
          <div style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
            <div
              style={{
                padding: "12px 16px 10px",
                color: "rgba(255,255,255,0.24)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              Live Updated Clips
            </div>
            {liveClips.slice(0, 6).map((clip, index) => (
              <div
                key={`live-${clip.id}`}
                style={{
                  borderTop: "1px solid rgba(255,255,255,0.03)",
                  contentVisibility: "auto",
                  containIntrinsicSize: "112px",
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
            ))}
          </div>
        )}

        {displayedResults.length === 0 ? (
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
