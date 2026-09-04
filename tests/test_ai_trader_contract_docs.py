from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AGENT_CONTRACT = (ROOT / "docs" / "AGENT_CONTRACT.md").read_text(encoding="utf-8")
TRADER_PLAYBOOK = (ROOT / "docs" / "SKILL_trader.md").read_text(encoding="utf-8")


class AITraderContractDocsTests(unittest.TestCase):
    def test_ai_is_the_discretionary_decision_owner(self) -> None:
        self.assertIn("AI-primary decision-authority cutover", AGENT_CONTRACT)
        self.assertIn("AI owns discretionary market decisions", AGENT_CONTRACT)
        self.assertIn("AI_DECISION_AUTHORITY=SHADOW", AGENT_CONTRACT)
        self.assertIn("AI owns\nthe discretionary market decision", TRADER_PLAYBOOK)

    def test_ai_may_author_a_complete_trade_decision(self) -> None:
        required_fields = (
            "pair",
            "side",
            "method",
            "vehicle",
            "entry",
            "TP",
            "SL",
            "allocation multiplier",
            "units",
        )
        authority_section = TRADER_PLAYBOOK.split("## Required current evidence", 1)[0]
        for field in required_fields:
            self.assertIn(field, authority_section)

        self.assertIn(
            "AI may choose `TRADE`, `WAIT`, or `REQUEST_EVIDENCE`",
            TRADER_PLAYBOOK,
        )
        self.assertIn("AI may reject the deterministic baseline", TRADER_PLAYBOOK)
        self.assertIn("trader-apply-market-read", TRADER_PLAYBOOK)
        self.assertIn("gpt-trader-decision", TRADER_PLAYBOOK)

    def test_live_execution_remains_a_separate_disabled_authority(self) -> None:
        self.assertIn("AI_ORDER_AUTHORITY=NONE", AGENT_CONTRACT)
        self.assertIn("AI_ORDER_AUTHORITY=NONE", TRADER_PLAYBOOK)
        self.assertIn("live_permission=false", TRADER_PLAYBOOK)
        self.assertIn("broker_mutation_allowed=false", TRADER_PLAYBOOK)
        self.assertIn("Do not call\n  OANDA", TRADER_PLAYBOOK)
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
