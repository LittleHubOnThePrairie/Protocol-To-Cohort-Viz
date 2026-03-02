"""CTR-XML (CDISC ODM) extractor for PTCV-19.

Parses EU Clinical Trials Register XML files (CDISC ODM extension)
natively using lxml. Produces TextBlock records from study metadata,
arm descriptions, and visit schedule entries.

No PDF fallback is attempted for CTR-XML inputs (per PTCV-19
Scenario 4 acceptance criteria).

Risk tier: MEDIUM — data pipeline component.

Regulatory references:
- ALCOA+ Traceable: source_sha256 in every output row
- ALCOA+ Contemporaneous: timestamp set just before artifact write
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import TextBlock

logger = logging.getLogger(__name__)

# CDISC ODM / EU-CTR namespace URIs encountered in practice
_ODM_NS = {
    "odm": "http://www.cdisc.org/ns/odm/v1.3",
    "ctr": "http://www.ema.europa.eu/clinicaltrials",
}


class CtrXmlExtractor:
    """Extracts study design metadata from a CTR-XML (CDISC ODM) file.

    The extractor produces TextBlock records covering:
    - GlobalVariables (StudyName, StudyDescription, ProtocolName)
    - StudyEventDef labels (visit schedule entries)
    - ArmDef labels and descriptions (treatment arms)
    - Any top-level StudyParameter annotations (free text)

    Table extraction is not applicable to CTR-XML; the tables.parquet
    artifact is written with zero rows when the source is CTR-XML.
    """

    def extract(
        self,
        xml_bytes: bytes,
        run_id: str,
        registry_id: str,
        source_sha256: str,
    ) -> tuple[list["TextBlock"], int]:
        """Parse CTR-XML bytes and return TextBlock records.

        Args:
            xml_bytes: Raw XML file contents.
            run_id: UUID4 for this extraction run.
            registry_id: Trial identifier.
            source_sha256: SHA-256 of the source XML file.

        Returns:
            Tuple of (text_blocks, page_count) where page_count is 0
            for non-page-based XML format.

        Raises:
            ValueError: If the bytes do not appear to be valid XML.
        """
        from lxml import etree  # type: ignore[import-untyped]

        try:
            root = etree.fromstring(xml_bytes)
        except etree.XMLSyntaxError as exc:
            raise ValueError(
                f"CTR-XML parse error for {registry_id}: {exc}"
            ) from exc

        blocks: list[TextBlock] = []
        block_index = 0

        def _add(text: str, block_type: str) -> None:
            nonlocal block_index
            stripped = (text or "").strip()
            if not stripped:
                return
            from .models import TextBlock

            blocks.append(
                TextBlock(
                    run_id=run_id,
                    source_registry_id=registry_id,
                    source_sha256=source_sha256,
                    page_number=0,
                    block_index=block_index,
                    text=stripped,
                    block_type=block_type,
                )
            )
            block_index += 1

        # ----- GlobalVariables ---------------------------------------
        gv: Any
        for gv in root.iter():
            local = etree.QName(gv.tag).localname if "}" in gv.tag else gv.tag
            if local == "StudyName":
                _add(gv.text or "", "heading")
            elif local == "StudyDescription":
                _add(gv.text or "", "paragraph")
            elif local == "ProtocolName":
                _add(gv.text or "", "heading")

        # ----- StudyEventDef (visit schedule) ------------------------
        sed: Any
        for sed in root.iter():
            local = etree.QName(sed.tag).localname if "}" in sed.tag else sed.tag
            if local == "StudyEventDef":
                name = sed.get("Name") or ""
                desc_elem = self._first_description(sed)
                _add(f"Visit: {name}", "list_item")
                if desc_elem is not None and desc_elem.text:
                    _add(str(desc_elem.text), "paragraph")

        # ----- ArmDef (treatment arms) -------------------------------
        arm: Any
        for arm in root.iter():
            local = etree.QName(arm.tag).localname if "}" in arm.tag else arm.tag
            if local == "ArmDef":
                name = arm.get("Name") or ""
                _add(f"Arm: {name}", "list_item")
                desc_elem = self._first_description(arm)
                if desc_elem is not None and desc_elem.text:
                    _add(str(desc_elem.text), "paragraph")

        if not blocks:
            logger.warning(
                "CtrXmlExtractor: no blocks extracted for %s — "
                "document may use non-standard namespace or structure.",
                registry_id,
            )

        return blocks, 0  # page_count = 0 for XML

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_description(element: Any) -> Any:
        """Return first child element whose local name is 'Description'.

        Args:
            element: lxml Element to search.

        Returns:
            First matching child element, or None.
        """
        from lxml import etree  # type: ignore[import-untyped]

        for child in element:
            local = etree.QName(child.tag).localname if "}" in child.tag else child.tag
            if local == "Description":
                return child
        return None
