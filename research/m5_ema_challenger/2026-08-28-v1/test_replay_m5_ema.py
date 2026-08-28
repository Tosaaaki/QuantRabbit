import importlib.util, json, pathlib, unittest

HERE=pathlib.Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("replay",HERE/"replay_m5_ema.py")
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads((HERE/"preregistration.json").read_text()); cls.data=M.load_inputs(cls.p); cls.sha=M.sha_file(HERE/"preregistration.json")

    def test_hash_binding_and_completed_chronology(self):
        for pair,rows in self.data.items():
            self.assertEqual(M.sha_file(pathlib.Path(self.p["inputs"]["files"][pair]["path"])),self.p["inputs"]["files"][pair]["sha256"])
            self.assertTrue(all(r["complete"] for r in rows)); self.assertTrue(all(rows[i]["_time"]<rows[i+1]["_time"] for i in range(len(rows)-1)))

    def test_no_cost_gate_and_next_bar_fill(self):
        for pair,rows in self.data.items():
            sig=M.make_signals(pair,rows,self.sha)
            self.assertGreater(len(sig),len(rows)*.99-20)
            self.assertTrue(all(s["fill_index"]==s["decision_index"]+1 for s in sig))
            self.assertEqual(len({s["signal_id"] for s in sig}),len(sig))

    def test_bid_ask_side_and_cost_arms_share_signal_set(self):
        pair="EUR_USD"; rows=self.data[pair][:200]; sig=M.make_signals(pair,rows,self.sha); tp=M.tuning_tp(pair,rows,sig,len(rows))
        eligible,trades,_=M.replay_config(pair,rows,sig,0,"A",6,tp)
        self.assertEqual({s["signal_id"] for s in eligible},{s["signal_id"] for s in sig})
        self.assertTrue(all(t["base_pips"]>t["adverse_pips"] for t in trades))
        self.assertTrue(all(abs((t["base_pips"]-t["adverse_pips"])-1.2)<1e-9 for t in trades))
        self.assertTrue(all((t["entry_executable"]>t["entry_raw"]) if t["side"]=="LONG" else (t["entry_executable"]<t["entry_raw"]) for t in trades))

    def test_finite_exit_terminal_liquidation_and_no_future(self):
        pair="USD_JPY"; rows=self.data[pair][:60]; sig=M.make_signals(pair,rows,self.sha); tp=M.tuning_tp(pair,rows,sig,len(rows))
        _,trades,_=M.replay_config(pair,rows,sig,0,"A",24,tp)
        self.assertTrue(all(t["exit_index"]>=t["entry_index"] for t in trades))
        self.assertTrue(all(t["age_bars"]<=24 for t in trades))
        self.assertTrue(any(t["exit_reason"]=="TERMINAL_LIQUIDATION" for t in trades))
        self.assertTrue(all(t["entry_index"]==next(s["fill_index"] for s in sig if s["signal_id"]==t["signal_id"]) for t in trades))
        base=M.exit_trade(pair,rows,{s["decision_index"]:s for s in sig},sig[0],"A",6,tp)
        changed=[dict(r) for r in rows]
        changed[-1]=dict(changed[-1],bid=dict(changed[-1]["bid"],c=999.0),ask=dict(changed[-1]["ask"],c=999.1))
        self.assertEqual(base,M.exit_trade(pair,changed,{s["decision_index"]:s for s in sig},sig[0],"A",6,tp))

    def test_tuning_tp_is_bounded_and_managed_exits_use_next_open(self):
        pair="EUR_USD"; rows=self.data[pair][:800]; split=600; sig=M.make_signals(pair,rows,self.sha)
        tp=M.tuning_tp(pair,rows,sig,split)
        changed=[dict(r) for r in rows]
        for i in range(split,len(changed)):
            changed[i]=dict(changed[i],bid=dict(changed[i]["bid"],c=9.0),ask=dict(changed[i]["ask"],c=9.1))
        self.assertEqual(tp,M.tuning_tp(pair,changed,sig,split))
        _,trades,_=M.replay_config(pair,rows,sig,0,"D",24,tp)
        managed=[t for t in trades if t["exit_reason"].endswith("NEXT_OPEN")]
        self.assertTrue(managed)
        for t in managed:
            r=rows[t["exit_index"]]
            self.assertEqual(t["exit_at"],"OPEN")
            self.assertAlmostEqual(t["exit_raw"],(r["bid"]["o"]+r["ask"]["o"])/2)
            self.assertAlmostEqual(t["exit_executable"],r["bid"]["o"] if t["side"]=="LONG" else r["ask"]["o"])

    def test_metrics_include_mtm_drawdown_month_start_and_ruin_gate(self):
        pair="USD_JPY"; rows=self.data[pair][:1000]; sig=M.make_signals(pair,rows,self.sha)
        tp=M.tuning_tp(pair,rows,sig,len(rows)); eligible,trades,skips=M.replay_config(pair,rows,sig,0,"D",6,tp)
        m=M.metrics(eligible,trades,skips,"adverse",self.data["USD_JPY"])
        self.assertEqual(m["drawdown_basis"],"completed_bar_portfolio_mtm_including_open_inventory")
        self.assertEqual(m["terminal_open_inventory"],0)
        self.assertIn("family_adjusted_lcb_pips",m)
        self.assertIn("break_even_roundtrip_cost_pips",m)

    def test_deterministic_rerun_and_packet_hash(self):
        a=M.main(True); rb=(HERE/"result.json").read_bytes(); pb=(HERE/"evidence_packet.json").read_bytes()
        b=M.main(True)
        self.assertEqual(a,b); self.assertEqual(rb,(HERE/"result.json").read_bytes()); self.assertEqual(pb,(HERE/"evidence_packet.json").read_bytes())
        packet=json.loads(pb); self.assertEqual(packet["result_sha256"],M.sha_file(HERE/"result.json")); self.assertEqual(packet["script_sha256"],M.sha_file(HERE/"replay_m5_ema.py"))

if __name__=="__main__": unittest.main()
