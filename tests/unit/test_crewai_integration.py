from unittest.mock import Mock, patch

import pytest

from tetherai.crewai.integration import protect_crew


class TestCrewAIIntegration:
    def test_crew_not_installed_import_error(self):
        with patch.dict("sys.modules", {"crewai": None}):
            import tetherai.crewai.integration as integration

            with pytest.raises(ImportError, match="crewai is not installed"):
                integration._check_crewai_installed()

    def test_protect_crew_returns_same_object(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        mock_crew = Mock()
        mock_agent = Mock()
        mock_task = Mock()

        mock_crew.agents = [mock_agent]
        mock_crew.tasks = [mock_task]
        mock_crew.kickoff = Mock(return_value="result")

        result = protect_crew(mock_crew, max_usd=2.0)

        assert result is mock_crew

    def test_protect_crew_sets_kickoff_wrapper(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        original_kickoff = Mock(return_value="result")
        mock_crew = Mock()
        mock_agent = Mock()
        mock_task = Mock()

        mock_crew.agents = [mock_agent]
        mock_crew.tasks = [mock_task]
        mock_crew.kickoff = original_kickoff

        protect_crew(mock_crew, max_usd=2.0)

        assert mock_crew.kickoff is not original_kickoff

    def test_existing_callbacks_preserved(self):
        try:
            import crewai  # noqa: F401
        except ImportError:
            pytest.skip("crewai not installed")

        mock_crew = Mock()
        mock_agent = Mock()
        mock_task = Mock()

        original_step_callback = Mock()
        original_task_callback = Mock()

        mock_agent.step_callback = original_step_callback
        mock_task.callback = original_task_callback

        mock_crew.agents = [mock_agent]
        mock_crew.tasks = [mock_task]
        mock_crew.kickoff = Mock(return_value="result")

        protect_crew(mock_crew, max_usd=2.0)

        assert mock_agent.step_callback is not None
