"""
FDSNChannelSelector — interactive channel picker for Jupyter notebooks

Usage
-----
    from es_notebook_utils import FDSNChannelSelector

    selector = FDSNChannelSelector()   # defaults to EarthScope FDSN
    selector.show()                    # renders the widget in the notebook

    # --- After selecting channels in the UI ---

    # List of (net, sta, loc, cha, start, end) tuples. start/end are
    # UTCDateTime objects clipped to the narrower of the channel's metadata
    # epoch and the search window.
    selector.selected
    selector.selected_inventory()      # ObsPy Inventory of selected channels

Setting initial search criteria
-------------------------------
    selector = FDSNChannelSelector(
        default_network="TA",
        default_station="*",
        default_location="*",
        default_channel="HH?",
        default_starttime="2024-01-01",
        default_endtime="2024-02-01",
    )

Disabling the map
-----------------
    selector = FDSNChannelSelector(show_map=False)

Other FDSN endpoints
--------------------
    selector = FDSNChannelSelector("https://service.iris.edu")
    selector = FDSNChannelSelector("IRIS")   # ObsPy shorthand also works

Requirements
------------
    pip install obspy ipywidgets ipyleaflet pandas
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import ipywidgets as widgets
import pandas as pd
from IPython.display import display as ipython_display, clear_output
from ipyleaflet import (
    LayerGroup, Map, Marker, MarkerCluster, Popup, basemaps,
)
from ipywidgets import HTML
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


PAGE_SIZE = 15


class FDSNChannelSelector:
    def __init__(
        self,
        client: str = "https://service.earthscope.org",
        default_network: str = "IU",
        default_station: str = "ANMO",
        default_location: str = "*",
        default_channel: str = "BH?",
        default_starttime: Optional[str] = None,
        default_endtime: Optional[str] = None,
        show_map: bool = True,
    ):
        self.client = Client(client)
        self.client_name = client
        self._show_map_enabled = show_map

        # State
        self.df: pd.DataFrame = pd.DataFrame()
        self.inventory = None  # full inventory from latest search
        self._search_start: Optional[UTCDateTime] = None
        self._search_end: Optional[UTCDateTime] = None
        self.page: int = 0
        self.selected_keys: set = set()  # keys are (net, sta, loc, cha)

        # Visual alignment: every "column 1" label gets the same width, so
        # the input boxes line up across rows regardless of label length.
        # Same idea for "column 2" (Time:) labels.
        LABEL_W = "90px"
        TIME_LABEL_W = "50px"
        INPUT_W = "260px"
        TIME_INPUT_W = "120px"
        label_style = {"description_width": LABEL_W}
        time_label_style = {"description_width": TIME_LABEL_W}

        # ----- Search controls -----
        self.net_in = widgets.Text(
            value=default_network, description="Network:",
            placeholder="e.g. IU,II  (* allowed)",
            style=label_style, layout=widgets.Layout(width=INPUT_W),
        )
        self.sta_in = widgets.Text(
            value=default_station, description="Station:",
            placeholder="e.g. ANMO,COLA",
            style=label_style, layout=widgets.Layout(width=INPUT_W),
        )
        self.loc_in = widgets.Text(
            value=default_location, description="Location:",
            placeholder="-- for blank, * for any",
            style=label_style, layout=widgets.Layout(width=INPUT_W),
        )
        self.cha_in = widgets.Text(
            value=default_channel, description="Channel:",
            placeholder="e.g. BH?,HH?",
            style=label_style, layout=widgets.Layout(width=INPUT_W),
        )

        # Time range — default: last 30 days through now (covers most "is this
        # station currently operating" queries while leaving room to widen).
        now = UTCDateTime()
        default_start = (
            UTCDateTime(default_starttime) if default_starttime else now - 30 * 86400
        )
        default_end = UTCDateTime(default_endtime) if default_endtime else now

        # Each time has two text widgets, both locale-independent:
        #   * Date  — YYYY-MM-DD
        #   * Time  — HH:MM:SS
        # Combined in _get_time() when the search runs.
        self.start_date = widgets.Text(
            value=default_start.datetime.strftime("%Y-%m-%d"),
            description="Start date:",
            placeholder="YYYY-MM-DD",
            style=label_style,
            layout=widgets.Layout(width=INPUT_W),
        )
        self.start_time = widgets.Text(
            value=default_start.datetime.strftime("%H:%M:%S"),
            description="Time:",
            placeholder="HH:MM:SS",
            style=time_label_style,
            layout=widgets.Layout(width=TIME_INPUT_W),
        )

        self.end_date = widgets.Text(
            value=default_end.datetime.strftime("%Y-%m-%d"),
            description="End date:",
            placeholder="YYYY-MM-DD",
            style=label_style,
            layout=widgets.Layout(width=INPUT_W),
        )
        self.end_time = widgets.Text(
            value=default_end.datetime.strftime("%H:%M:%S"),
            description="Time:",
            placeholder="HH:MM:SS",
            style=time_label_style,
            layout=widgets.Layout(width=TIME_INPUT_W),
        )

        # No sync wiring needed — date and time text fields are read directly
        # when
        # building the search query.

        self.search_btn = widgets.Button(
            description="Search", button_style="primary", icon="search"
        )
        self.search_btn.on_click(self._on_search)

        self.status = widgets.HTML(value="")

        # ----- Results table -----
        self.table_out = widgets.Output()
        self.prev_btn = widgets.Button(description="◀ Prev", disabled=True)
        self.next_btn = widgets.Button(description="Next ▶", disabled=True)
        self.page_label = widgets.HTML(value="")
        self.prev_btn.on_click(lambda _: self._change_page(-1))
        self.next_btn.on_click(lambda _: self._change_page(+1))

        self.select_page_btn = widgets.Button(description="Select page")
        self.select_all_btn = widgets.Button(description="Select all")
        self.clear_btn = widgets.Button(description="Clear all")
        self.select_page_btn.on_click(self._select_page)
        self.select_all_btn.on_click(self._select_all)
        self.clear_btn.on_click(self._clear_all)

        # ----- Map -----
        self.map = Map(
            center=(20, 0),
            zoom=2,
            basemap=basemaps.OpenStreetMap.Mapnik,
            scroll_wheel_zoom=True,
        )
        self.map.layout.height = "450px"
        # Cache markers by (Net, Sta) — building a Marker is expensive over
        # the comm channel, so reuse them across map updates. Diff the desired
        # set against this cache on each update; only build new markers.
        self._marker_cache: dict = {}        # (Net, Sta) -> Marker
        # The current layer attached to the map (either a MarkerCluster or
        # a LayerGroup of Markers). One layer = one comm message to swap.
        self._station_layer = None

        self.auto_update_toggle = widgets.ToggleButton(
            value=True,
            description="Auto update",
            tooltip="When on, the map updates automatically as you select/deselect channels",
            icon="refresh",
            button_style="success",
        )
        # When auto-update is flipped on, immediately render the current
        # selection so users see the map populate without further clicks.
        self.auto_update_toggle.observe(
            lambda c: self._update_map() if c["new"] else None,
            names="value",
        )

        self.cluster_toggle = widgets.ToggleButton(
            value=True,
            description="Cluster",
            tooltip="Group nearby stations into clusters (uncheck to show all markers individually)",
            icon="object-group",
        )
        self.cluster_toggle.observe(
            lambda c: self._update_map(),
            names="value",
        )

        self.selection_count = widgets.HTML(value="<b>0</b> channels selected")

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def _get_time(self, which: str) -> UTCDateTime:
        """Combine YYYY-MM-DD date + HH:MM:SS time into a UTCDateTime."""
        date_w = self.start_date if which == "start" else self.end_date
        time_w = self.start_time if which == "start" else self.end_time

        date_str = (date_w.value or "").strip()
        if not date_str:
            raise ValueError(f"{which.title()} date is empty")

        time_str = (time_w.value or "00:00:00").strip()
        parts = time_str.split(":")
        if len(parts) == 2:  # accept HH:MM
            time_str = f"{time_str}:00"

        # UTCDateTime parses "YYYY-MM-DDTHH:MM:SS" and will raise on bad input,
        # which _on_search() catches and surfaces in the status bar.
        return UTCDateTime(f"{date_str}T{time_str}")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def show(self):
        search_row1 = widgets.HBox([self.net_in, self.sta_in])
        search_row2 = widgets.HBox([self.loc_in, self.cha_in])
        start_row = widgets.HBox([self.start_date, self.start_time])
        end_row = widgets.HBox([self.end_date, self.end_time])
        button_row = widgets.HBox([self.search_btn, self.status])

        page_controls = widgets.HBox(
            [self.prev_btn, self.page_label, self.next_btn,
             self.select_page_btn, self.select_all_btn, self.clear_btn,
             self.selection_count]
        )

        results_box = widgets.VBox([page_controls, self.table_out])

        ui_children = [
            widgets.HTML(f"<h3>FDSN Channel Selector — <code>{self.client_name}</code></h3>"),
            search_row1, search_row2, start_row, end_row, button_row,
            widgets.HTML("<hr>"),
            results_box,
        ]

        if self._show_map_enabled:
            map_box = widgets.VBox([
                widgets.HBox([self.auto_update_toggle, self.cluster_toggle]),
                self.map,
            ])
            ui_children += [widgets.HTML("<hr>"), map_box]

        ui = widgets.VBox(ui_children)
        ipython_display(ui)
    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _on_search(self, _):
        self.status.value = "<i>Searching…</i>"
        try:
            starttime = self._get_time("start")
            endtime = self._get_time("end")
        except Exception as e:
            self.status.value = f"<span style='color:red'>Time error: {e}</span>"
            return
        try:
            inv = self.client.get_stations(
                network=self.net_in.value or "*",
                station=self.sta_in.value or "*",
                location=self.loc_in.value or "*",
                channel=self.cha_in.value or "*",
                starttime=starttime,
                endtime=endtime,
                level="channel",
            )
        except Exception as e:
            self.status.value = f"<span style='color:red'>Error: {e}</span>"
            return

        self.inventory = inv
        # Remember the search window — used by selected() to clip channel
        # epochs against what the user actually asked for.
        self._search_start = starttime
        self._search_end = endtime

        rows = []
        for net in inv:
            for sta in net:
                for cha in sta:
                    desc = ""
                    if cha.sensor and cha.sensor.description:
                        desc = cha.sensor.description
                    rows.append({
                        "Net": net.code,
                        "Sta": sta.code,
                        "Loc": cha.location_code or "--",
                        "Cha": cha.code,
                        "Lat": round(cha.latitude, 4),
                        "Lon": round(cha.longitude, 4),
                        "Elev": round(cha.elevation, 1),
                        "SR": cha.sample_rate,
                        "Description": desc,
                        # Display columns (date only)
                        "Start": str(cha.start_date)[:10] if cha.start_date else "",
                        "End": str(cha.end_date)[:10] if cha.end_date else "",
                        # Full-precision UTCDateTime epochs for selected()
                        "_StartUTC": cha.start_date,
                        "_EndUTC": cha.end_date,
                    })
        self.df = pd.DataFrame(rows)
        self.page = 0
        self.status.value = f"<b>{len(self.df)}</b> channels found"
        # New result set — drop any cached markers and clear the map. Stations
        # in the new search may have different metadata or coordinates, so the
        # cache from the previous search isn't safe to reuse.
        self._marker_cache = {}
        self._detach_station_layer()
        self._render_table()

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------
    def _key(self, row) -> Tuple[str, str, str, str]:
        return (row["Net"], row["Sta"], row["Loc"], row["Cha"])

    def _render_table(self):
        """Full re-render: builds all widgets fresh. Called on search/pagination."""
        self._row_buttons: dict = {}  # row_idx -> list[Button] for the selection cell + others
        with self.table_out:
            clear_output(wait=True)
            if self.df.empty:
                print("No results. Adjust criteria and search.")
                self._update_pagination()
                return

            start = self.page * PAGE_SIZE
            end = start + PAGE_SIZE
            page_df = self.df.iloc[start:end]

            row_widgets = []
            header = widgets.HBox([
                widgets.HTML("<b>Sel</b>", layout=widgets.Layout(width="40px")),
                widgets.HTML("<b>Net</b>", layout=widgets.Layout(width="50px")),
                widgets.HTML("<b>Sta</b>", layout=widgets.Layout(width="70px")),
                widgets.HTML("<b>Loc</b>", layout=widgets.Layout(width="50px")),
                widgets.HTML("<b>Cha</b>", layout=widgets.Layout(width="60px")),
                widgets.HTML("<b>Lat</b>", layout=widgets.Layout(width="80px")),
                widgets.HTML("<b>Lon</b>", layout=widgets.Layout(width="80px")),
                widgets.HTML("<b>SR</b>", layout=widgets.Layout(width="70px")),
                widgets.HTML("<b>Description</b>", layout=widgets.Layout(width="240px")),
                widgets.HTML("<b>Start</b>", layout=widgets.Layout(width="100px")),
                widgets.HTML("<b>End</b>", layout=widgets.Layout(width="100px")),
            ])
            row_widgets.append(header)

            for row_idx, row in page_df.iterrows():
                key = self._key(row)
                is_selected = key in self.selected_keys
                full_desc = row["Description"] or ""
                short_desc = (
                    full_desc if len(full_desc) <= 32 else full_desc[:30] + "…"
                )
                cells = [
                    ("✓" if is_selected else "☐", "40px", ""),
                    (row["Net"], "50px", ""),
                    (row["Sta"], "70px", ""),
                    (row["Loc"], "50px", ""),
                    (row["Cha"], "60px", ""),
                    (str(row["Lat"]), "80px", ""),
                    (str(row["Lon"]), "80px", ""),
                    (str(row["SR"]), "70px", ""),
                    (short_desc, "240px", full_desc),
                    (row["Start"], "100px", ""),
                    (row["End"], "100px", ""),
                ]
                bg = "#cce5ff" if is_selected else "white"
                row_buttons = []
                for label, width, tooltip in cells:
                    btn = widgets.Button(
                        description=str(label),
                        tooltip=tooltip,
                        layout=widgets.Layout(
                            width=width, height="28px", padding="0px",
                            margin="0px", border="1px solid #e0e0e0",
                        ),
                        style={"button_color": bg, "font_weight": "normal"},
                    )
                    btn.on_click(self._make_row_toggle(key, row_idx))
                    row_buttons.append(btn)
                self._row_buttons[row_idx] = row_buttons
                row_widgets.append(widgets.HBox(row_buttons))

            ipython_display(widgets.VBox(row_widgets))
            self._update_pagination()
            self._update_count()

    def _refresh_row_style(self, row_idx):
        """Patch a single row's appearance without rebuilding any widgets.

        This is the fast path used on click toggles. We mutate the existing
        Button widgets' style and the first cell's description, which only
        sends a few small traitlet updates over the comm channel — vs the
        ~165 widget creations a full _render_table call costs.
        """
        buttons = self._row_buttons.get(row_idx)
        if not buttons:
            return
        row = self.df.iloc[row_idx]
        is_selected = self._key(row) in self.selected_keys
        bg = "#cce5ff" if is_selected else "white"
        for btn in buttons:
            btn.style.button_color = bg
        # Update the ✓/☐ indicator in the first cell
        buttons[0].description = "✓" if is_selected else "☐"

    def _make_row_toggle(self, key, row_idx):
        """Return a click handler that toggles selection of the given row."""
        def _toggle(_btn):
            if key in self.selected_keys:
                self.selected_keys.discard(key)
            else:
                self.selected_keys.add(key)
            # Fast path: only patch the affected row's styles.
            self._refresh_row_style(row_idx)
            self._update_count()
            self._maybe_auto_update_map()
        return _toggle

    def _maybe_auto_update_map(self):
        """Refresh the map if the user has auto-update enabled."""
        if not self._show_map_enabled:
            return
        if self.auto_update_toggle.value:
            self._update_map()

    def _update_pagination(self):
        n_pages = max(1, (len(self.df) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.page_label.value = (
            f"&nbsp;Page <b>{self.page + 1}</b> / {n_pages}&nbsp;"
        )
        self.prev_btn.disabled = self.page <= 0
        self.next_btn.disabled = self.page >= n_pages - 1

    def _update_count(self):
        self.selection_count.value = (
            f"&nbsp;<b>{len(self.selected_keys)}</b> channels selected"
        )

    def _change_page(self, delta):
        self.page += delta
        self._render_table()

    def _select_page(self, _):
        start = self.page * PAGE_SIZE
        end = start + PAGE_SIZE
        for _, row in self.df.iloc[start:end].iterrows():
            self.selected_keys.add(self._key(row))
        self._render_table()
        self._maybe_auto_update_map()

    def _select_all(self, _):
        for _, row in self.df.iterrows():
            self.selected_keys.add(self._key(row))
        self._render_table()
        self._maybe_auto_update_map()

    def _clear_all(self, _):
        self.selected_keys.clear()
        self._render_table()
        # Always drop the layer when clearing — the user explicitly emptied
        # the selection, so leaving stale pins would be misleading regardless
        # of the auto-update setting.
        self._detach_station_layer()

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def _update_map(self):
        """Render selected stations on the map.

        Performance notes:
          * Markers are cached by (Net, Sta) so toggling selections doesn't
            recreate marker widgets the user has seen before.
          * The map gets exactly one composite layer (MarkerCluster or
            LayerGroup) attached. Swapping it is one comm message regardless
            of how many markers it contains.
          * The previous implementation removed N markers individually,
            which was the source of the slowness on clear-then-redraw.
        """
        # Compute the set of stations to display
        if not self.selected_keys:
            self._detach_station_layer()
            return

        keys = list(zip(self.df["Net"], self.df["Sta"], self.df["Loc"], self.df["Cha"]))
        mask = [k in self.selected_keys for k in keys]
        sel_df = self.df[mask]
        if sel_df.empty:
            self._detach_station_layer()
            return

        sta_df = sel_df.drop_duplicates(subset=["Net", "Sta"])
        wanted_stations = set(zip(sta_df["Net"], sta_df["Sta"]))

        # Channel list per (Net, Sta) for the popup — single groupby pass.
        chan_lists = (
            sel_df.assign(_loccha=sel_df["Loc"] + "." + sel_df["Cha"])
                  .groupby(["Net", "Sta"])["_loccha"]
                  .apply(lambda s: ", ".join(sorted(set(s))))
                  .to_dict()
        )

        # Build only the markers we don't already have cached.
        for _, row in sta_df.iterrows():
            station_key = (row["Net"], row["Sta"])
            if station_key in self._marker_cache:
                # Update popup in case the channel list changed for this
                # station (e.g. user added more channels at the same station).
                m = self._marker_cache[station_key]
                if isinstance(m.popup, HTML):
                    m.popup.value = (
                        f"<b>{row['Net']}.{row['Sta']}</b><br>"
                        f"{row['Lat']:.4f}, {row['Lon']:.4f}<br>"
                        f"Channels: {chan_lists[station_key]}"
                    )
                continue
            popup_html = (
                f"<b>{row['Net']}.{row['Sta']}</b><br>"
                f"{row['Lat']:.4f}, {row['Lon']:.4f}<br>"
                f"Channels: {chan_lists[station_key]}"
            )
            m = Marker(location=(row["Lat"], row["Lon"]), draggable=False)
            m.popup = HTML(value=popup_html)
            self._marker_cache[station_key] = m

        # Collect just the markers we want to show right now
        markers = [self._marker_cache[k] for k in wanted_stations]

        # Build the new composite layer
        if self.cluster_toggle.value:
            new_layer = MarkerCluster(markers=markers)
        else:
            new_layer = LayerGroup(layers=markers)

        # Atomically swap the station layer. substitute_layer sends one
        # comm message; remove + add would send two. Either way it's much
        # cheaper than removing N markers individually.
        if self._station_layer is None:
            self.map.add_layer(new_layer)
        else:
            try:
                self.map.substitute_layer(self._station_layer, new_layer)
            except Exception:
                # Fallback for older ipyleaflet versions without substitute_layer
                self.map.remove_layer(self._station_layer)
                self.map.add_layer(new_layer)
        self._station_layer = new_layer

        # Auto-fit bounds
        lats = sta_df["Lat"].tolist()
        lons = sta_df["Lon"].tolist()
        if len(lats) == 1:
            self.map.center = (lats[0], lons[0])
            self.map.zoom = 6
        else:
            self.map.fit_bounds([
                [min(lats), min(lons)],
                [max(lats), max(lons)],
            ])

    def _detach_station_layer(self):
        """Remove the current station layer from the map (if any)."""
        if self._station_layer is None:
            return
        try:
            self.map.remove_layer(self._station_layer)
        except Exception:
            pass
        self._station_layer = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def selected(self) -> List[Tuple[str, str, str, str, UTCDateTime, UTCDateTime]]:
        """List of selected channels as (net, sta, loc, cha, start, end) tuples.

        The start and end times are the narrowest of the channel's metadata
        epoch and the search window — i.e. the time range over which the user
        could actually request data for this channel given both what they
        asked for and what the channel was operating.

        - If the channel's start_date is None (open-ended past), the search
          start is used.
        - If the channel's end_date is None (still operating), the search end
          is used.
        - If no search has run yet, the channel epoch is returned as-is and
          unbounded sides are returned as None.
        """
        if self.df.empty or not self.selected_keys:
            return []

        results = []
        # Build a lookup from key -> (start, end) row data once
        keys = list(zip(
            self.df["Net"], self.df["Sta"], self.df["Loc"], self.df["Cha"]
        ))
        starts = self.df["_StartUTC"].tolist()
        ends = self.df["_EndUTC"].tolist()

        for key, ch_start, ch_end in zip(keys, starts, ends):
            if key not in self.selected_keys:
                continue
            # Clip against the search window
            if self._search_start is not None and ch_start is not None:
                start = max(ch_start, self._search_start)
            elif ch_start is not None:
                start = ch_start
            else:
                start = self._search_start  # may be None

            if self._search_end is not None and ch_end is not None:
                end = min(ch_end, self._search_end)
            elif ch_end is not None:
                end = ch_end
            else:
                end = self._search_end  # may be None

            results.append((*key, start, end))
        return sorted(results)

    def selected_inventory(self):
        """Return an ObsPy Inventory subset matching current selections."""
        if self.inventory is None or not self.selected_keys:
            return None
        nslc_set = self.selected_keys
        # Build via select() — call once per selection then merge
        from obspy import Inventory
        result = Inventory(networks=[], source=self.inventory.source)
        for net, sta, loc, cha in nslc_set:
            sub = self.inventory.select(
                network=net, station=sta, location=loc or "", channel=cha
            )
            result += sub
        return result
        