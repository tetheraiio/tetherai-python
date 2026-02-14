from unittest.mock import Mock, patch

import pytest

from tetherai.crewai.integration import ProtectedCrew, protect_crew


class TestCrewAIIntegration:
    def test_crew_not_installed_import_error(self):
        with patch.dict("sys.modules", {"crewai": None}):
            import tetherai.crewai.integration as integration

            with pytest.raises(ImportError, match="crewai is not installed"):
                integration._check_crewai_installed()

    def test_protect_crew_returns_protected_crew(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        mock_crew = Mock()
        mock_crew.kickoff = Mock(return_value="result")

        result = protect_crew(mock_crew, max_usd=2.0)

        assert isinstance(result, ProtectedCrew)
        assert result._crew is mock_crew

    def test_protected_crew_kickoff_is_wrapped(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        original_kickoff = Mock(return_value="result")
        mock_crew = Mock()
        mock_crew.kickoff = original_kickoff

        protected = protect_crew(mock_crew, max_usd=2.0)

        assert protected.kickoff is not original_kickoff

    def test_protected_crew_proxies_other_attributes(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        mock_crew = Mock()
        mock_crew.kickoff = Mock(return_value="result")
        mock_crew.agents = [Mock()]
        mock_crew.tasks = [Mock()]

        protected = protect_crew(mock_crew, max_usd=2.0)

        assert protected.agents == mock_crew.agents
        assert protected.tasks == mock_crew.tasks
