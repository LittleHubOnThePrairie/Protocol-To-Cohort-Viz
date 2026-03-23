"""Tests for Vision API figure captioning (PTCV-226).

Tests caption generation, prompt selection, graceful degradation,
and cost tracking. API calls are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from ptcv.extraction.figure_captioner import (
    FigureCaption,
    FigureCaptioner,
    CaptionResult,
    _FIGURE_TYPE_PROMPTS,
)
from ptcv.extraction.figure_detector import (
    BoundingBox,
    DetectedFigure,
)


def _make_figure(
    page: int = 1,
    fig_type: str = "generic",
    caption: str = "",
) -> DetectedFigure:
    return DetectedFigure(
        page_number=page,
        bbox=BoundingBox(50, 100, 500, 600),
        figure_type_hint=fig_type,
        caption=caption,
    )


class TestFigureCaption:
    """Tests for FigureCaption dataclass."""

    def test_has_caption_true(self):
        cap = FigureCaption(
            figure=_make_figure(),
            caption="Study design with 2:1 randomization",
        )
        assert cap.has_caption

    def test_has_caption_false(self):
        cap = FigureCaption(figure=_make_figure())
        assert not cap.has_caption

    def test_from_existing(self):
        cap = FigureCaption(
            figure=_make_figure(),
            caption="Existing",
            from_existing=True,
        )
        assert cap.from_existing


class TestCaptionResult:
    """Tests for CaptionResult dataclass."""

    def test_empty(self):
        result = CaptionResult()
        assert result.figures_captioned == 0
        assert result.figures_skipped == 0
        assert result.api_calls_made == 0

    def test_counts(self):
        result = CaptionResult(captions=[
            FigureCaption(figure=_make_figure(), caption="Good"),
            FigureCaption(figure=_make_figure(), caption=""),
            FigureCaption(figure=_make_figure(), caption="Also good"),
        ])
        assert result.figures_captioned == 2
        assert result.figures_skipped == 1


class TestFigureTypePrompts:
    """Tests for prompt template coverage."""

    def test_all_types_have_prompts(self):
        expected = {
            "study_design", "consort", "kaplan_meier",
            "pk_profile", "forest_plot", "dose_response", "generic",
        }
        assert set(_FIGURE_TYPE_PROMPTS.keys()) == expected

    def test_prompts_non_empty(self):
        for fig_type, prompt in _FIGURE_TYPE_PROMPTS.items():
            assert len(prompt) > 20, f"Prompt for {fig_type} too short"


class TestFigureCaptionerNoApi:
    """Tests when API key is not set."""

    def test_no_api_key_returns_empty_captions(self):
        """Test graceful degradation without API key."""
        captioner = FigureCaptioner()

        with patch.dict("os.environ", {}, clear=True):
            result = captioner.caption_figures(
                figures=[_make_figure()],
                pdf_bytes=b"%PDF-stub",
            )

        assert result.figures_captioned == 0
        assert result.figures_skipped == 1
        assert result.api_calls_made == 0

    def test_existing_caption_preserved(self):
        """Test figures with existing captions skip API."""
        captioner = FigureCaptioner()
        fig = _make_figure(caption="Figure 1: Study Design Schema")

        result = captioner.caption_figures(
            figures=[fig],
            pdf_bytes=b"%PDF-stub",
        )

        assert result.figures_captioned == 1
        assert result.api_calls_made == 0
        assert result.captions[0].caption == "Figure 1: Study Design Schema"
        assert result.captions[0].from_existing

    def test_empty_figures_list(self):
        """Test empty input returns empty result."""
        captioner = FigureCaptioner()
        result = captioner.caption_figures([], b"%PDF")
        assert result.figures_captioned == 0


class TestFigureCaptionerWithApi:
    """Tests with mocked API."""

    def _mock_client(self, caption_text: str, tokens: int = 500):
        """Build a mock Anthropic client."""
        mock_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = caption_text
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(
            input_tokens=tokens - 100,
            output_tokens=100,
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_api_call_produces_caption(self):
        """Test successful Vision API captioning."""
        captioner = FigureCaptioner()
        captioner._client = self._mock_client(
            "2:1 randomization to Drug A vs placebo",
            tokens=800,
        )

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=b"fake_png_bytes",
        ):
            result = captioner.caption_figures(
                figures=[_make_figure(fig_type="study_design")],
                pdf_bytes=b"%PDF",
            )

        assert result.figures_captioned == 1
        assert result.api_calls_made == 1
        assert result.total_token_cost == 800
        assert "randomization" in result.captions[0].caption

    def test_crop_failure_produces_empty_caption(self):
        """Test empty caption when image cropping fails."""
        captioner = FigureCaptioner()
        captioner._client = self._mock_client("should not appear")

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=None,
        ):
            result = captioner.caption_figures(
                figures=[_make_figure()],
                pdf_bytes=b"%PDF",
            )

        assert result.figures_captioned == 0
        assert result.api_calls_made == 0

    def test_api_error_returns_empty_caption(self):
        """Test graceful handling of API errors."""
        captioner = FigureCaptioner()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        captioner._client = mock_client

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=b"fake_png",
        ):
            result = captioner.caption_figures(
                figures=[_make_figure()],
                pdf_bytes=b"%PDF",
            )

        assert result.figures_captioned == 0
        assert result.api_calls_made == 1
        assert result.total_token_cost == 0

    def test_multiple_figures_cost_accumulated(self):
        """Test token cost summed across multiple figures."""
        captioner = FigureCaptioner()
        captioner._client = self._mock_client("Caption", tokens=500)

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=b"fake_png",
        ):
            result = captioner.caption_figures(
                figures=[
                    _make_figure(page=1),
                    _make_figure(page=5),
                    _make_figure(page=12),
                ],
                pdf_bytes=b"%PDF",
            )

        assert result.figures_captioned == 3
        assert result.api_calls_made == 3
        assert result.total_token_cost == 1500

    def test_mixed_existing_and_api(self):
        """Test mix of existing captions and API-generated ones."""
        captioner = FigureCaptioner()
        captioner._client = self._mock_client("API caption")

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=b"fake_png",
        ):
            result = captioner.caption_figures(
                figures=[
                    _make_figure(caption="Existing caption"),
                    _make_figure(),  # needs API
                ],
                pdf_bytes=b"%PDF",
            )

        assert result.figures_captioned == 2
        assert result.api_calls_made == 1
        assert result.captions[0].from_existing
        assert not result.captions[1].from_existing

    def test_figure_type_selects_correct_prompt(self):
        """Test that figure type hint selects the right prompt."""
        captioner = FigureCaptioner()
        mock_client = self._mock_client("CONSORT description")
        captioner._client = mock_client

        with patch(
            "ptcv.extraction.figure_captioner.crop_figure_image",
            return_value=b"fake_png",
        ):
            captioner.caption_figures(
                figures=[_make_figure(fig_type="consort")],
                pdf_bytes=b"%PDF",
            )

        # Check the prompt in the API call
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]
        text_content = [c for c in content if c["type"] == "text"][0]
        assert "CONSORT" in text_content["text"]
        assert "number screened" in text_content["text"]
