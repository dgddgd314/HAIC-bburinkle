from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

STATE_DIR = Path(__file__).resolve().parent / "state"
REQUEST = STATE_DIR / "request.json"
PROGRESS = STATE_DIR / "progress.json"

MISSIONS = ("1 go to (x, y) · 2 go to the <colour> cone · 3 cross the cliff course · "
            "4 deliver the pencil to its room · 5 pencil into the holder · 6 score a goal")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def busy() -> bool:
    request, progress = read_json(REQUEST), read_json(PROGRESS) or {}
    if not request:
        return False
    return progress.get("id") != request.get("id") or progress.get("state") in ("planning", "running")


def show_record(number: int, step: str, record: dict) -> None:
    verdict = record.get("verdict", "N/A")
    text = (f"{number}. {step} -- mission {record.get('mission')} {record.get('label', '')}: "
            f"**{verdict}** ({record.get('seconds', 0):.1f} s)")
    (st.success if verdict == "PASS" else st.error)(text)
    rooms = record.get("rooms")
    lines = []
    if rooms:
        lines += [f"rooms  truth : {rooms.get('truth')}", f"       answer: {rooms.get('answer')}",
                  f"       {rooms.get('correct')}/{rooms.get('total')} rooms right"]
    lines += [f"{c.get('name')}: {c.get('summary')}" for c in record.get("checks") or []
              if c.get("name") != "rooms"]
    reported = record.get("reported") or {}
    lines.append(f"reported {reported.get('status')}: {reported.get('detail')}")
    st.code("\n".join(lines), language="text")


def show_progress() -> None:
    request, progress = read_json(REQUEST), read_json(PROGRESS)
    if request and (not progress or progress.get("id") != request.get("id")):
        st.info(f"Sent: '{request.get('instruction')}' -- waiting for main.py "
                "(python main.py --llm-web) to take it ...")
        time.sleep(2.0)
        st.rerun()
    if not progress:
        return
    state = progress.get("state", "?")
    st.subheader(f"Instruction: {progress.get('instruction', '')}")
    if progress.get("error"):
        st.error(f"Your planner failed: {progress['error']}")
    plan = progress.get("plan") or []
    if state == "planning":
        st.info("custom/planner/planner.py is planning ...")
    elif plan:
        st.write("Plan: " + " → ".join(plan))
    records = {r.get("step"): r for r in progress.get("records", [])}
    for number, step in enumerate(plan, 1):
        if number in records:
            show_record(number, step, records[number])
        elif state == "running" and number == len(records) + 1:
            st.write(f"{number}. {step} -- running ...")
        else:
            st.write(f"{number}. {step} -- {'waiting' if state == 'running' else 'not run'}")
    if progress.get("total"):
        total = progress["total"]
        st.caption(f"{total.get('passed')}/{total.get('missions')} passed · "
                   f"{total.get('seconds', 0):.1f} s from the instruction to the last verdict")
    st.caption(f"state: {state} · results file: {progress.get('results_file')}")
    if state in ("planning", "running"):
        time.sleep(2.0)
        st.rerun()


def main() -> None:
    st.title("G1 mission instruction")
    st.markdown("**Missions:** " + MISSIONS)
    st.markdown("One instruction, the missions in the order you want them, with mission 1's (x, y) "
                "and mission 2's colour.")
    with st.form("instruction"):
        text = st.text_area("Instruction:", placeholder="Go to the blue cone, then score a goal, "
                                                        "and finally walk to (14.2, 7.7).")
        submitted = st.form_submit_button("Submit")
    if submitted and text.strip():
        if busy():
            st.warning("The previous instruction is still running; wait until it has finished.")
        else:
            write_json(REQUEST, {"id": str(time.time_ns()), "instruction": text.strip()})
    show_progress()


if __name__ == "__main__":
    main()
