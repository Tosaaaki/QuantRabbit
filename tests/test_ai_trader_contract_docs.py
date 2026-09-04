from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AGENT_CONTRACT = (ROOT / "docs" / "AGENT_CONTRACT.md").read_text(encoding="utf-8")
TRADER_PLAYBOOK = (ROOT / "docs" / "SKILL_trader.md").read_text(encoding="utf-8")


class AITraderContractDocsTests(unittest.TestCase):
    def test_ai_is_the_discretionary_decision_owner(self) -> None:
        self.assertIn("AI-primary decision runtime", AGENT_CONTRACT)
        self.assertIn("AI owns the discretionary market decision", TRADER_PLAYBOOK)
        self.assertNotIn("AI_DECISION_AUTHORITY=SHADOW", AGENT_CONTRACT)

    def test_ai_may_author_a_complete_trade_decision(self) -> None:
        required_fields = (
            "pair",
            "side",
            "method",
            "vehicle",
            "entry",
            "TP",
            "SL",
            "units",
        )
        authority_section = TRADER_PLAYBOOK.split("## Intraday cycle", 1)[0]
        for field in required_fields:
            self.assertIn(field, authority_section)

        self.assertIn("`ENTER`, `WAIT`, and `REQUEST_EVIDENCE`", TRADER_PLAYBOOK)
        self.assertIn("ai_trader_hotpath.py", TRADER_PLAYBOOK)
        self.assertIn("ai_trader_runtime.py accept", TRADER_PLAYBOOK)
        self.assertIn("There is no fixed 1,000-unit", TRADER_PLAYBOOK)
        self.assertIn("or allocation multiplier", TRADER_PLAYBOOK)

    def test_live_execution_uses_a_separate_gateway_authority(self) -> None:
        self.assertIn("QR_AI_ORDER_AUTHORITY=LIVE", AGENT_CONTRACT)
        self.assertIn("QR_AI_ORDER_AUTHORITY=LIVE", TRADER_PLAYBOOK)
        self.assertIn("RiskEngine", TRADER_PLAYBOOK)
        self.assertIn("LiveOrderGateway", TRADER_PLAYBOOK)
        self.assertIn("Do not call OANDA", TRADER_PLAYBOOK)
        self.assertIn("`NO_TOUCH`", TRADER_PLAYBOOK)

    def test_retired_blanket_ban_is_not_in_the_active_playbook(self) -> None:
        forbidden = (
            "It is not an order trader",
            "AI must not create or select an order action",
            "AI must not choose or alter direction",
            "Any request for order, cancel, close, direction",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, TRADER_PLAYBOOK)


if __name__ == "__main__":
    unittest.main()
