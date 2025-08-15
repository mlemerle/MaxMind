# ----- Stroop -----
def page_stroop():
    page_header("Stroop")
    st.caption("Tap the **ink color**, not the word.")

    # Adaptive default ISI
    def_idx = adaptive_suggest_index("stroop")
    isi_ms = st.selectbox("Stimulus duration (ms)", STROOP_GRID, index=def_idx)
    trials = st.selectbox("Trials", [20, 30, 50], index=1)

    if "stroop" not in st.session_state:
        st.session_state["stroop"] = None

    if st.button("Start"):
        st.session_state["stroop"] = {
            "i": 0, "correct": 0, "trials": trials, "isi": isi_ms,
            "words": ["RED","GREEN","BLUE","YELLOW"],
            "colors": {"RED":"#ff4d4d","GREEN":"#40c463","BLUE":"#4da3ff","YELLOW":"#ffd33d"},
            "current": None
        }
        _stroop_next()

    stp = st.session_state["stroop"]
    if stp and stp["current"]:
        word, ink = stp["current"]
        st.markdown(
            f"<div style='font-size:72px;text-align:center;color:{stp['colors'][ink]}'>{word}</div>",
            unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        for label, col in zip(stp["words"], [c1, c2, c3, c4]):
            if col.button(label):
                if label == ink:
                    stp["correct"] += 1
                # ISI timing: simulate with sleep then next
                time.sleep(stp["isi"]/1000.0)
                _stroop_next()

def _stroop_next():
    stp = st.session_state["stroop"]
    if not stp:
        return
    stp["i"] += 1
    if stp["i"] > stp["trials"]:
        acc = round(stp["correct"] / stp["trials"] * 100, 1)
        st.success(f"Done. Accuracy: {acc}%  ({stp['correct']}/{stp['trials']})")
        S()["stroopHistory"].append({"date": today_iso(), "trials": stp["trials"], "acc": acc, "isi": stp["isi"]})
        # Adaptive update
        level_idx = STROOP_GRID.index(stp["isi"]) if stp["isi"] in STROOP_GRID else adaptive_suggest_index("stroop")
        adaptive_update("stroop", level_idx, accuracy=acc/100.0)
        save_state()
        st.session_state["stroop"] = None
        st.rerun()
        return
    w = random.choice(stp["words"])
    c = random.choice(stp["words"])
    stp["current"] = (w, c)

# ----- Complex Span -----
def page_complex_span():
    page_header("Complex Span")
    st.caption("Remember letters **in order**, while verifying simple equations between letters (dual task).")

    # Adaptive default set-size
    def_idx = adaptive_suggest_index("complex_span")
    set_size = st.selectbox("Set size (letters to recall)", CSPAN_GRID, index=def_idx)
    equations_per_item = 1  # one verification between each letter

    if "cspan" not in st.session_state:
        st.session_state["cspan"] = None

    if st.button("Start"):
        letters = [random.choice("BCDFGHJKLMNPQRSTVWXYZ") for _ in range(set_size)]
        # Generate simple equation items (a±b=?), with truth flag
        eqs = []
        for _ in range(set_size * equations_per_item):
            a, b = random.randint(2,9), random.randint(2,9)
            op = random.choice(["+", "−"])
            true_val = a + b if op == "+" else a - b
            # Create a statement maybe correct, maybe off by 1-2
            if random.random() < 0.5:
                shown = true_val
                truth = True
            else:
                delta = random.choice([1,2,-1,-2])
                shown = true_val + delta
                truth = False
            eqs.append((a, op, b, shown, truth))
        st.session_state["cspan"] = {
            "letters": letters, "eqs": eqs, "i": 0,
            "proc_correct": 0, "proc_total": 0,
            "phase": "letters",  # letters -> verify -> recall
            "set_size": set_size
        }
        _cspan_next()

    cs = st.session_state["cspan"]
    if cs:
        if cs["phase"] == "letters":
            st.info(f"Remember: **{cs['letters'][cs['i']]}**")
            time.sleep(1.0)  # show for 1s
            cs["i"] += 1
            if cs["i"] >= cs["set_size"]:
                cs["phase"] = "verify"
                cs["i"] = 0
            st.rerun()

        elif cs["phase"] == "verify":
            a, op, b, shown, truth = cs["eqs"][cs["i"]]
            st.write(f"Is this true? **{a} {op} {b} = {shown}**")
            c1, c2 = st.columns(2)
            if c1.button("True"):
                if truth: cs["proc_correct"] += 1
                cs["proc_total"] += 1
                cs["i"] += 1
                if cs["i"] >= len(cs["eqs"]):
                    cs["phase"] = "recall"
                st.rerun()
            if c2.button("False"):
                if not truth: cs["proc_correct"] += 1
                cs["proc_total"] += 1
                cs["i"] += 1
                if cs["i"] >= len(cs["eqs"]):
                    cs["phase"] = "recall"
                st.rerun()

        elif cs["phase"] == "recall":
            ans = st.text_input("Type the letters in order (no spaces)", key="cspan_recall")
            if st.button("Submit recall"):
                guess = [ch.upper() for ch in ans.strip()]
                correct_positions = sum(1 for i,ch in enumerate(guess[:cs["set_size"]]) if ch == cs["letters"][i])
                recall_acc = correct_positions / cs["set_size"]
                proc_acc = (cs["proc_correct"] / max(1, cs["proc_total"])) if cs["proc_total"] else 0.0
                composite = (recall_acc + proc_acc) / 2.0
                st.success(f"Recall: {correct_positions}/{cs['set_size']}  • Verify: {cs['proc_correct']}/{cs['proc_total']}  → Composite acc: {round(composite*100,1)}%")
                # Adaptive update
                level_idx = CSPAN_GRID.index(cs["set_size"])
                adaptive_update("complex_span", level_idx, accuracy=composite)
                save_state()
                st.session_state["cspan"] = None
                if st.button("Run again"):
                    st.rerun()

def _cspan_next():
    # Helper just to kick the first letter display
    pass

# ----- Go/No-Go -----
def page_gng():
    page_header("Go / No-Go")
    st.caption("Press **GO** for Go stimuli; do **nothing** for No-Go. Measures response inhibition.")

    def_idx = adaptive_suggest_index("gng")
    isi = st.selectbox("ISI (ms)", GNG_GRID, index=def_idx)
    trials = st.selectbox("Trials", [40, 60, 80], index=1)
    p_nogo = 0.20  # fixed no-go probability

    if "gng" not in st.session_state:
        st.session_state["gng"] = None

    if st.button("Start"):
        seq = []
        for _ in range(trials):
            if random.random() < p_nogo:
                seq.append(("NO_GO", "X"))  # show an 'X' for no-go
            else:
                seq.append(("GO", random.choice("BCDFGHJKLMNPQRSTVWXYZ")))
        st.session_state["gng"] = {
            "seq": seq, "isi": isi, "i": 0,
            "hits": 0, "misses": 0, "fa": 0,
            "last_seen_index": -1, "done": False
        }
        st.rerun()

    g = st.session_state["gng"]
    if g:
        placeholder = st.empty()
        if not g["done"]:
            if g["i"] < len(g["seq"]):
                stim_type, stim_val = g["seq"][g["i"]]
                with placeholder:
                    color = "#40c463" if stim_type == "GO" else "#ff4d4d"
                    st.markdown(
                        f"<div style='font-size:72px;text-align:center;color:{color}'>{stim_val}</div>",
                        unsafe_allow_html=True
                    )
                # Window to respond
                g["last_seen_index"] = g["i"]
                start = now_ts()
                # Give user an opportunity to click during ISI (best-effort in Streamlit)
                btn = st.button("GO", key=f"go_{g['i']}")
                if btn:
                    # If current is GO, it's a hit; if NO_GO, it's a false alarm
                    if stim_type == "GO":
                        g["hits"] += 1
                    else:
                        g["fa"] += 1
                # Wait till ISI elapses then move on; if GO and user did not press, it's a miss
                time.sleep(g["isi"]/1000.0)
                if stim_type == "GO" and not btn:
                    g["misses"] += 1
                g["i"] += 1
                st.rerun()
            else:
                g["done"] = True

        if g["done"]:
            go_total = sum(1 for t,_ in g["seq"] if t == "GO")
            nogo_total = len(g["seq"]) - go_total
            hit_rate = g["hits"] / max(1, go_total)
            fa_rate  = g["fa"]  / max(1, nogo_total)
            # Balanced accuracy proxy
            composite = (hit_rate + (1.0 - fa_rate)) / 2.0
            st.success(f"Go hits {g['hits']}/{go_total}, Misses {g['misses']}; No-Go false alarms {g['fa']}/{nogo_total}.")
            st.info(f"Composite accuracy ≈ {round(composite*100,1)}%")
            # Adaptive update
            level_idx = GNG_GRID.index(g["isi"])
            adaptive_update("gng", level_idx, accuracy=composite)
            save_state()
            if st.button("Reset"):
                st.session_state["gng"] = None
                st.rerun()

# ----- Mental Math -----
def page_mm():
    page_header("Mental Math")
    mode = st.selectbox("Mode", ["Percent", "Fraction→Decimal", "Quick Ops", "Fermi"])
    duration_min = st.selectbox("Duration (min)", [2, 3, 5], index=1)
    tol = st.selectbox("Tolerance", ["Exact", "±5%", "±10%"], index=0)

    if "mm" not in st.session_state:
        st.session_state["mm"] = None

    def gen_problem():
        r = random.randint
        if mode == "Percent":
            p, base = r(5, 35), r(40, 900)
            return (f"What is {p}% of {base}?", round(base * p / 100, 2))
        if mode == "Fraction→Decimal":
            n, d = r(1, 9), r(2, 19)
            return (f"Convert {n}/{d} to decimal (4 dp).", round(n / d, 4))
        if mode == "Quick Ops":
            a, b, op = r(12, 99), r(6, 24), random.choice(["×", "÷", "+", "−"])
            ans = a * b if op == "×" else round(a / b, 3) if op == "÷" else a + b if op == "+" else a - b
            return (f"{a} {op} {b} = ?", ans)
        # Fermi
        prompts = [
            ("How many minutes in 6 weeks?", 6*7*24*60),
            ("How many seconds in 3 hours?", 3*3600),
            ("Sheets in 2 cm stack at 0.1 mm/page?", 200),
            ("Liters in 10m×2m×1m pool?", 20000),
        ]
        return random.choice(prompts)

    if st.button("Start"):
        end = now_ts() + duration_min * 60
        st.session_state["mm"] = {"end": end, "score": 0, "total": 0, "cur": gen_problem()}
        st.rerun()

    mm = st.session_state["mm"]
    if mm:
        left = int(mm["end"] - now_ts())
        st.metric("Time", timer_text(left))
        st.write(f"**Problem:** {mm['cur'][0]}")
        ans = st.text_input("Answer", key="mm_ans")
        if st.button("Submit"):
            user = ans.strip()
            if user:
                try:
                    user_val = float(user)
                    correct = False
                    truth = mm["cur"][1]
                    if tol == "Exact":
                        correct = abs(user_val - truth) < 1e-9
                    else:
                        pct = 0.05 if tol == "±5%" else 0.10
                        correct = abs(user_val - truth) <= pct * max(1.0, abs(truth))
                    mm["total"] += 1
                    if correct:
                        mm["score"] += 1
                        st.success("Correct ✓")
                    else:
                        st.error(f"Answer: {truth}")
                except ValueError:
                    st.warning("Enter a number.")
            mm["cur"] = gen_problem()
            st.rerun()

        if left <= 0:
            acc = round((mm["score"] / max(1, mm["total"])) * 100, 1)
            st.success(f"Done. Correct: {mm['score']} / {mm['total']} (Acc {acc}%)")
            S()["mmHistory"].append({"date": today_iso(), "mode": mode, "acc": acc})
            save_state()
            st.session_state["mm"] = None
            if st.button("Restart"):
                st.rerun()

# ----- Writing -----
def page_writing():
    page_header("Writing Sprint (12 min)")
    if "w" not in st.session_state:
        st.session_state["w"] = None

    # Enhanced prompts using concepts from your card deck
    prompts = [
        "Explain Bayes' theorem using a medical test example and why base rates matter.",
        "Describe a multipolar trap and provide a real-world example (climate change, arms race, etc.).",
        "What is Moloch? How does it manifest in modern society and what can be done about it?",
        "Explain the concept of feedback loops using examples from both positive and negative feedback.",
        "What is Goodhart's Law and how does it apply to modern metrics and KPIs?",
        "Describe the difference between System 1 and System 2 thinking with practical examples.",
        "What is emergence and how does it apply to complex systems?",
        "Explain instrumental convergence and why it matters for AI safety.",
        "What makes a good explanation according to David Deutsch's epistemology?",
        "How do leverage points in systems thinking help us create change effectively?",
    ]
    
    colA, colB = st.columns([1,2])
    with colA:
        ptxt = st.text_area("Prompt", value=random.choice(prompts), height=100)
        if st.button("Start 12-min"):
            st.session_state["w"] = {"end": now_ts() + 12*60, "prompt": ptxt, "text": ""}
            st.rerun()
        if st.session_state["w"]:
            left = int(st.session_state["w"]["end"] - now_ts())
            st.metric("Time", timer_text(left))
    with colB:
        if st.session_state["w"]:
            txt = st.text_area("Draft (write without stopping)", value=st.session_state["w"]["text"], height=300, key="w_draft")
            st.session_state["w"]["text"] = txt
            if now_ts() >= st.session_state["w"]["end"]:
                st.success("Time! Review your draft.")
                if st.button("Save session"):
                    S()["writingSessions"].append({
                        "date": today_iso(),
                        "prompt": st.session_state["w"]["prompt"],
                        "text": st.session_state["w"]["text"]
                    })
                    save_state()
                    st.session_state["w"] = None
                    st.rerun()

# ----- Forecasts -----
def page_forecasts():
    page_header("Forecast Journal")
    q = st.text_input("Question", placeholder="Will X happen by YYYY-MM-DD?")
    p = st.number_input("Probability (%)", min_value=1, max_value=99, value=60)
    due = st.date_input("Due date")
    notes = st.text_area("Notes", height=80)
    if st.button("Add"):
        if q.strip():
            S()["forecasts"].append({
                "id": new_id(), "q": q.strip(), "p": int(p),
                "due": due.isoformat(), "notes": notes.strip(),
                "created": today_iso(), "resolved": None, "outcome": None
            })
            save_state()
            st.success("Added.")
            st.rerun()

    items = sorted(S()["forecasts"], key=lambda x: x["due"])
    for f in items:
        with st.container(border=True):
            st.write(f"**{f['q']}**")
            st.caption(f"p={f['p']}% • due {f['due']} • created {f['created']}")
            if f.get("notes"):
                st.write(f"_{f['notes']}_")
            c1, c2, c3 = st.columns(3)
            if not f["resolved"]:
                if c1.button("Resolve TRUE", key=f"t_{f['id']}"):
                    f["resolved"] = today_iso(); f["outcome"] = 1; save_state(); st.rerun()
                if c2.button("Resolve FALSE", key=f"f_{f['id']}"):
                    f["resolved"] = today_iso(); f["outcome"] = 0; save_state(); st.rerun()
            if c3.button("Delete", key=f"d_{f['id']}"):
                S()["forecasts"] = [x for x in S()["forecasts"] if x["id"] != f["id"]]
                save_state(); st.rerun()

    resolved = [f for f in S()["forecasts"] if f["resolved"] is not None]
    if resolved:
        st.markdown("### Calibration & Brier decomposition")
        if st.button("Show calibration"):
            reliability_curve(resolved)
            brier, rel, res, unc = brier_decomposition(resolved)
            st.write(f"**Brier**: {brier:.3f}  •  Reliability: {rel:.3f}  •  Resolution: {res:.3f}  •  Uncertainty: {unc:.3f}")
            st.caption("Lower Brier is better. Reliability↓ (calibration) and Resolution↑ (discrimination) are desirable; Uncertainty depends on base rate.")

def reliability_curve(resolved: List[Dict[str, Any]]):
    # Bin into deciles by forecast p
    bins = [(i/10.0, (i+1)/10.0) for i in range(10)]
    pts = []
    for lo, hi in bins:
        xs = [f for f in resolved if lo <= f["p"]/100.0 < hi] if hi < 1 else [f for f in resolved if lo <= f["p"]/100.0 <= hi]
        if xs:
            mean_p = sum(f["p"]/100.0 for f in xs)/len(xs)
            freq = sum(f["outcome"] for f in xs)/len(xs)
            pts.append((mean_p, freq, len(xs)))
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--")
    if pts:
        ax.scatter([x for x,_,_ in pts],[y for _,y,_ in pts])
        for x,y,n in pts:
            ax.text(x, y, f"{n}", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Forecast calibration")
    st.pyplot(fig)

def brier_decomposition(resolved: List[Dict[str, Any]]) -> Tuple[float,float,float,float]:
    """Murphy (1973) decomposition: Brier = Reliability - Resolution + Uncertainty"""
    y = [f["outcome"] for f in resolved]
    p = [f["p"]/100.0 for f in resolved]
    n = len(y)
    # Overall Brier
    brier = sum((pi - yi)**2 for pi, yi in zip(p,y))/n
    # Group by deciles
    buckets: Dict[int, List[int]] = {}
    for i,(pi,yi) in enumerate(zip(p,y)):
        b = min(9, int(pi*10))
        buckets.setdefault(b, []).append(i)
    rel = 0.0
    res = 0.0
    ybar = sum(y)/n
    for idxs in buckets.values():
        ni = len(idxs)
        pbar_i = sum(p[i] for i in idxs)/ni
        ybar_i = sum(y[i] for i in idxs)/ni
        rel += ni/n * (pbar_i - ybar_i)**2
        res += ni/n * (ybar_i - ybar)**2
    unc = ybar * (1 - ybar)
    # Brier ≈ rel - res + unc (floating error possible)
    return brier, rel, res, unc

# ----- Enhanced Argument Map -----
def page_argmap():
    page_header("Argument Map")
    
    # Suggest concepts from your card deck for argument practice
    suggested_topics = [
        "Interleaving beats blocking for durable learning",
        "Utilitarianism is the best moral framework",
        "AI alignment is solvable with current approaches", 
        "Systems thinking is superior to reductionist thinking",
        "Bayesian reasoning should be taught in schools",
        "Moloch problems require coordination solutions",
        "Emergent properties can't be predicted from components",
        "Fallibilism is the most rational epistemic stance"
    ]
    
    thesis = st.selectbox("Choose a thesis (or write your own below):", 
                         [""] + suggested_topics, 
                         index=0)
    if not thesis:
        thesis = st.text_input("Custom thesis", value="Interleaving beats blocking for durable learning.")
    
    pros = st.text_area("Reasons (one per line)").strip().splitlines()
    cons = st.text_area("Objections (one per line)").strip().splitlines()
    rebs = st.text_area("Rebuttals (one per line)").strip().splitlines()
    
    if st.button("Render Argument Map"):
        dot = Digraph(engine="dot")
        dot.attr(rankdir="TB", bgcolor="transparent")
        dot.node("T", thesis, shape="box", style="rounded,filled", fillcolor="#1e2433", fontcolor="white")
        
        for i, r in enumerate([p for p in pros if p.strip()]):
            nid = f"P{i}"
            dot.node(nid, r, shape="box", style="rounded,filled", fillcolor="#1b2a1d", fontcolor="white")
            dot.edge(nid, "T", color="green")
            
        for i, c in enumerate([c for c in cons if c.strip()]):
            nid = f"C{i}"
            dot.node(nid, c, shape="box", style="rounded,filled", fillcolor="#331e1e", fontcolor="white")
            dot.edge("T", nid, color="red")
            
        for i, rb in enumerate([r for r in rebs if r.strip()]):
            nid = f"R{i}"
            dot.node(nid, rb, shape="box", style="rounded,filled", fillcolor="#262233", fontcolor="white")
            if cons:
                dot.edge(nid, "C0", color="blue")
            else:
                dot.edge(nid, "T", color="blue")
                
        st.graphviz_chart(dot)

# ----- Settings -----
def page_settings():
    page_header("Settings & Backup")
    s = S()["settings"]
    nl = st.number_input("Daily new cards", min_value=0, max_value=50, value=s["newLimit"])
    rl = st.number_input("Review limit", min_value=10, max_value=500, value=s["reviewLimit"])
    if st.button("Save settings"):
        s["newLimit"] = int(nl); s["reviewLimit"] = int(rl); save_state(); st.success("Saved.")

    st.markdown("#### Export / Import")
    st.download_button("Export JSON", data=export_json(), file_name="max_mind_trainer_backup.json")
    up = st.file_uploader("Import JSON", type=["json"])
    if up and st.button("Import now"):
        import_json(up.read().decode("utf-8"))

    st.markdown("#### Card Statistics")
    cards = S()["cards"]
    total_cards = len(cards)
    learned_cards = len([c for c in cards if not c.get("new")])
    
    # Tag statistics
    tag_counts = {}
    for card in cards:
        for tag in card.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    st.write(f"**Total cards**: {total_cards}")
    st.write(f"**Learned cards**: {learned_cards}")
    st.write(f"**New cards**: {total_cards - learned_cards}")
    
    if tag_counts:
        st.markdown("**Cards by tag**:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"• {tag}: {count} cards")

    st.markdown("#### Seed deck management")
    with st.expander("Reset to default cards"):
        st.warning("This will replace all current cards with the default seed. Your progress will be lost!")
        if st.button("⚠️ Reset to Default Cards"):
            S()["cards"] = [asdict(c) for c in make_cards(DEFAULT_SEED)]
            save_state()
            st.success("Cards reset to default seed.")
            st.rerun()

# ========== Router ==========
PAGES = [
    "Dashboard",
    "Spaced Review", 
    "N-Back",
    "Stroop",
    "Complex Span",
    "Go/No-Go",
    "Mental Math",
    "Writing",
    "Forecasts",
    "Argument Map",
    "Settings",
]

st.set_page_config(page_title="Max Mind Trainer", page_icon="🧠", layout="centered")
with st.sidebar:
    st.title("Max Mind Trainer")
    st.caption("Enhanced Cognitive Training")
    st.session_state.setdefault("page", "Dashboard")
    st.session_state["page"] = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]))

page = st.session_state["page"]
if page == "Dashboard": page_dashboard()
elif page == "Spaced Review": page_review()
elif page == "N-Back": page_nback()
elif page == "Stroop": page_stroop()
elif page == "Complex Span": page_complex_span()
elif page == "Go/No-Go": page_gng()
elif page == "Mental Math": page_mm()
elif page == "Writing": page_writing()
elif page == "Forecasts": page_forecasts()
elif page == "Argument Map": page_argmap()
elif page == "Settings": page_settings()
