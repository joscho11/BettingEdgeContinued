"""Registry of TikTok videos surfaced in the Film Room tab.

To add a video after posting it:
  1. Append an entry to VIDEOS below (slug, title, subtitle, date, tiktok_url, video_id,
     breakdown_file).
  2. Drop its in-depth breakdown as markdown in  video_breakdowns/<breakdown_file>.
The `video_id` is the number at the end of the TikTok URL (.../video/<id>).

ORDER DOES NOT MATTER HERE. film_room.py sorts episodes by `date`, newest first
(default selection). This list stays append-only. `date` is ISO (YYYY-MM-DD) and is the publish
date shown on the player; an entry without one still renders, it just carries no date line and
sorts to the end.
"""

# Optional channel intro. None until a replacement is posted. The old Welcome video
# (id 7660252294327307550) is gone: it was outdated and its ATS figure was wrong.
# To restore a Start here control, set this to a dict with title, date, tiktok_url,
# video_id, and optional about/blurb/subtitle. film_room.py already knows how to
# render it.
INTRO_VIDEO = None

# Player / topic videos. Each gets an embed + a click-to-open written breakdown.
# `archived: True` + `archive_note` adds a compact "📼 Archived: why?" pop-out to the
# card (the note + Draft Board cross-link live inside it; see film_room.render_film_room).
VIDEOS = [
    {
        "slug": "brian-thomas-jr",
        "title": "The Market Is Wrong About Brian Thomas Jr.",
        "date": "2026-07-07",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7660252626046553374",
        "video_id": "7660252626046553374",
        "breakdown_file": "brian_thomas_jr.md",
        "archived": True,
        "archive_note": (
            "📼 Archived. Posted July 7, 2026, before my validation work "
            "finished. This video makes a call about one player using a model "
            "I've since retired. When testing finished, what held up were "
            "group-level patterns and calibrated ranges, never claims about "
            "individual players, and this video doesn't reflect how I work now. "
            "It stays up, "
            "unedited, as part of the record. For what I publish today: the "
            "Draft Board page."
        ),
    },
    {
        "slug": "makai-lemon",
        "title": "Makai Lemon: Rookie Receiver Profile",
        "subtitle": "2026 · WR, Philadelphia Eagles",
        "date": "2026-07-30",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7668110810039717151",
        "video_id": "7668110810039717151",
        "breakdown_file": "makai_lemon.md",
    },
    {
        "slug": "bijan-robinson-jahmyr-gibbs",
        "title": "Bijan Robinson vs. Jahmyr Gibbs",
        "subtitle": "2025 season review · RB, Atlanta / Detroit",
        "date": "2026-08-02",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7669558168984309022",
        "video_id": "7669558168984309022",
        "breakdown_file": "bijan_robinson_jahmyr_gibbs.md",
    },
    {
        "slug": "how-to-leverage-adp-guide",
        "title": "How to Leverage ADP: Guide",
        "subtitle": "2026 · when two projections both disagree with the market",
        "date": "2026-08-04",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7670323446793915679",
        "video_id": "7670323446793915679",
        "breakdown_file": "how_to_leverage_adp_guide.md",
    },
    {
        "slug": "how-to-leverage-adp-qb",
        "title": "How to Leverage ADP: QB Edition",
        "subtitle": "2026 · the five QBs with a gap",
        "date": "2026-08-05",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7670687538364845342",
        "video_id": "7670687538364845342",
        "breakdown_file": "how_to_leverage_adp_qb.md",
    },
    {
        "slug": "how-to-leverage-adp-te",
        "title": "How to Leverage ADP: TE Edition",
        "subtitle": "2026 · the seven TEs with a gap",
        "date": "2026-08-06",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7671059325892349214",
        "video_id": "7671059325892349214",
        "breakdown_file": "how_to_leverage_adp_te.md",
    },
    {
        "slug": "how-to-leverage-adp-rb",
        "title": "How to Leverage ADP: RB Edition",
        "subtitle": "2026 · one backfield, six rounds apart",
        "date": "2026-08-08",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7671785941031324958",
        "video_id": "7671785941031324958",
        "breakdown_file": "how_to_leverage_adp_rb.md",
    },
    {
        "slug": "how-to-leverage-adp-wr",
        "title": "How to Leverage ADP: WR Edition",
        "subtitle": "2026 · Flowers +7, Moore −11",
        "date": "2026-08-11",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7672900851412847902",
        "video_id": "7672900851412847902",
        "breakdown_file": "how_to_leverage_adp_wr.md",
    },
    {
        "slug": "draft-order",
        "title": "Does draft order actually decide your season?",
        "subtitle": "2026 · 3,641 public Sleeper snake leagues",
        "date": "2026-08-13",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7673639176264355102",
        "video_id": "7673639176264355102",
        "breakdown_file": "draft_order.md",
    },
    {
        "slug": "qb-te-draft-timing",
        "title": "When Should You Draft a QB and TE?",
        "subtitle": "2018-2025 · 1,422 public 1QB Sleeper leagues",
        "date": "2026-08-15",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7674314953565670687",
        "video_id": "7674314953565670687",
        "breakdown_file": "qb_te_draft_timing.md",
    },
    {
        "slug": "league-history",
        "title": "Who's the best manager in your league?",
        "subtitle": "2026 · Sleeper League History walkthrough",
        "date": "2026-08-16",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7674717547266133278",
        "video_id": "7674717547266133278",
        "breakdown_file": "league_history.md",
    },
    {
        "slug": "jefferson-deep-dive",
        "title": "Justin Jefferson's Fantasy Outlook 2026",
        "subtitle": "2026 · WR, Minnesota",
        "date": "2026-08-17",
        "tiktok_url": "https://www.tiktok.com/@joschoanalytics/video/7675129111659957534",
        "video_id": "7675129111659957534",
        "breakdown_file": "jefferson_deep_dive.md",
    },
]
