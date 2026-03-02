"""Define-XML v2.1 generator for CDISC SDTM Trial Design domains.

Produces a minimal but structurally valid Define-XML v2.1 document that:
- References each of the 5 XPT datasets (TS, TA, TE, TV, TI)
- Lists all variables present in each dataset with their attributes
- Declares the ODM / Define-XML v2.1 namespace
- Can be stored via StorageGateway alongside the XPT artifacts

Risk tier: MEDIUM — regulatory submission artefact.

Regulatory references:
- CDISC Define-XML v2.1 Specification
- FDA Study Data Technical Conformance Guide v5.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

try:
    from lxml import etree as ET  # type: ignore[import-untyped]
    _LXML = True
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[assignment]
    _LXML = False

import pandas as pd


# ---------------------------------------------------------------------------
# Namespace declarations
# ---------------------------------------------------------------------------

_ODM_NS = "http://www.cdisc.org/ns/odm/v1.3"
_DEF_NS = "http://www.cdisc.org/ns/def/v2.1"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

_NS = {
    "def": _DEF_NS,
    "odm": _ODM_NS,
    "xsi": _XSI_NS,
}

# SDTM variable attribute defaults per domain/variable
_VAR_ATTRS: dict[str, dict[str, dict[str, str]]] = {
    "TS": {
        "STUDYID":  {"label": "Study Identifier",         "type": "Char", "length": "20"},
        "DOMAIN":   {"label": "Domain Abbreviation",       "type": "Char", "length": "2"},
        "TSSEQ":    {"label": "Sequence Number",           "type": "Num",  "length": "8"},
        "TSPARMCD": {"label": "Trial Summary Parameter CD","type": "Char", "length": "8"},
        "TSPARM":   {"label": "Trial Summary Parameter",   "type": "Char", "length": "40"},
        "TSVAL":    {"label": "Parameter Value",           "type": "Char", "length": "200"},
        "TSVALCD":  {"label": "Parameter Value Code",      "type": "Char", "length": "20"},
        "TSVALNF":  {"label": "Reason Parameter Value Null","type": "Char", "length": "8"},
        "TSVALUNIT":{"label": "Parameter Value Unit",      "type": "Char", "length": "40"},
    },
    "TA": {
        "STUDYID":  {"label": "Study Identifier",         "type": "Char", "length": "20"},
        "DOMAIN":   {"label": "Domain Abbreviation",       "type": "Char", "length": "2"},
        "ARMCD":    {"label": "Planned Arm Code",          "type": "Char", "length": "20"},
        "ARM":      {"label": "Description of Planned Arm","type": "Char", "length": "40"},
        "TAETORD":  {"label": "Planned Order of Elements", "type": "Num",  "length": "8"},
        "ETCD":     {"label": "Element Code",              "type": "Char", "length": "8"},
        "ELEMENT":  {"label": "Description of Element",    "type": "Char", "length": "40"},
        "TABRANCH": {"label": "Branch",                    "type": "Char", "length": "200"},
        "TATRANS":  {"label": "Transition Rule",           "type": "Char", "length": "200"},
        "EPOCH":    {"label": "Epoch",                     "type": "Char", "length": "40"},
    },
    "TE": {
        "STUDYID":  {"label": "Study Identifier",         "type": "Char", "length": "20"},
        "DOMAIN":   {"label": "Domain Abbreviation",       "type": "Char", "length": "2"},
        "ETCD":     {"label": "Element Code",              "type": "Char", "length": "8"},
        "ELEMENT":  {"label": "Description of Element",    "type": "Char", "length": "40"},
        "TESTRL":   {"label": "Rule for Start of Element", "type": "Char", "length": "200"},
        "TEENRL":   {"label": "Rule for End of Element",   "type": "Char", "length": "200"},
        "TEDUR":    {"label": "Planned Duration of Element","type": "Char", "length": "20"},
    },
    "TV": {
        "STUDYID":  {"label": "Study Identifier",         "type": "Char", "length": "20"},
        "DOMAIN":   {"label": "Domain Abbreviation",       "type": "Char", "length": "2"},
        "VISITNUM": {"label": "Visit Number",              "type": "Num",  "length": "8"},
        "VISIT":    {"label": "Visit Name",                "type": "Char", "length": "40"},
        "VISITDY":  {"label": "Planned Study Day of Visit","type": "Num",  "length": "8"},
        "TVSTRL":   {"label": "Visit Start Rule",          "type": "Num",  "length": "8"},
        "TVENRL":   {"label": "Visit End Rule",            "type": "Num",  "length": "8"},
        "TVTIMY":   {"label": "Planned Visit Time of Day", "type": "Char", "length": "8"},
        "TVTIM":    {"label": "Planned Clock Time of Visit","type": "Char", "length": "20"},
        "TVENDY":   {"label": "Planned End Day of Visit",  "type": "Char", "length": "20"},
    },
    "TI": {
        "STUDYID":  {"label": "Study Identifier",         "type": "Char", "length": "20"},
        "DOMAIN":   {"label": "Domain Abbreviation",       "type": "Char", "length": "2"},
        "IETESTCD": {"label": "Inclusion/Exclusion Test Short Name","type":"Char","length":"8"},
        "IETEST":   {"label": "Inclusion/Exclusion Criterion","type": "Char", "length": "200"},
        "IECAT":    {"label": "Inclusion/Exclusion Category","type": "Char", "length": "20"},
        "IESCAT":   {"label": "Inclusion/Exclusion Sub-Category","type":"Char","length":"40"},
        "TIRL":     {"label": "Trial Inclusion/Exclusion Criterion Rule","type":"Char","length":"200"},
        "TIVERS":   {"label": "Protocol Version",          "type": "Char", "length": "20"},
    },
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DefineXmlGenerator:
    """Generates a Define-XML v2.1 document for SDTM trial design domains.

    Args:
        study_name: Human-readable study name for the ODM header.
    """

    def __init__(self, study_name: str = "") -> None:
        self._study_name = study_name

    def generate(
        self,
        datasets: dict[str, pd.DataFrame],
        studyid: str,
        source_xpt_sha256s: dict[str, str] | None = None,
    ) -> bytes:
        """Generate Define-XML v2.1 as UTF-8 encoded bytes.

        Variable names in each DataFrame must match actual XPT columns.

        Args:
            datasets: Mapping from domain name (e.g. "TS") to DataFrame.
            studyid: STUDYID for the ODM Study element.
            source_xpt_sha256s: Optional mapping domain → XPT sha256 for
                provenance annotation in ItemGroupDef comments.

        Returns:
            UTF-8 encoded XML bytes of the Define-XML document.
        [PTCV-22 Scenario: Define-XML references each XPT dataset]
        """
        source_sha256s = source_xpt_sha256s or {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # Build root ODM element with proper namespace declarations.
        # lxml requires nsmap for xmlns:* declarations; stdlib uses
        # register_namespace().
        if _LXML:
            nsmap = {None: _ODM_NS, "def": _DEF_NS, "xsi": _XSI_NS}
            root = ET.Element(f"{{{_ODM_NS}}}ODM", nsmap=nsmap)  # type: ignore[call-arg]
        else:
            import xml.etree.ElementTree as _stdlib_ET
            _stdlib_ET.register_namespace("", _ODM_NS)
            _stdlib_ET.register_namespace("def", _DEF_NS)
            _stdlib_ET.register_namespace("xsi", _XSI_NS)
            root = ET.Element(f"{{{_ODM_NS}}}ODM")

        root.set(
            f"{{{_XSI_NS}}}schemaLocation",
            "http://www.cdisc.org/ns/odm/v1.3 "
            "http://www.cdisc.org/ns/odm/v1.3/ODM1-3-2.xsd",
        )
        root.set("ODMVersion", "1.3.2")
        root.set("FileType", "Snapshot")
        root.set("FileOID", f"DefinitionFile.{studyid}")
        root.set("CreationDateTime", now)
        root.set("Originator", "PTCV-22 SdtmService")
        root.set("SourceSystem", "PTCV")
        root.set("SourceSystemVersion", "1.0")

        # Study element
        study = ET.SubElement(root, f"{{{_ODM_NS}}}Study", attrib={"OID": studyid})

        # GlobalVariables
        gv = ET.SubElement(study, f"{{{_ODM_NS}}}GlobalVariables")
        ET.SubElement(gv, f"{{{_ODM_NS}}}StudyName").text = (
            self._study_name or studyid
        )
        ET.SubElement(gv, f"{{{_ODM_NS}}}StudyDescription").text = (
            "CDISC SDTM Trial Design Domains"
        )
        ET.SubElement(gv, f"{{{_ODM_NS}}}ProtocolName").text = studyid

        # MetaDataVersion
        mdv = ET.SubElement(
            study,
            f"{{{_ODM_NS}}}MetaDataVersion",
            attrib={
                "OID": f"MDV.{studyid}",
                "Name": f"{studyid} Define-XML v2.1",
                f"{{{_DEF_NS}}}DefineVersion": "2.1.0",
                f"{{{_DEF_NS}}}StandardName": "SDTM",
                f"{{{_DEF_NS}}}StandardVersion": "1.7",
            },
        )

        # Collect all ItemDef OIDs
        itemdef_added: set[str] = set()

        for domain, df in datasets.items():
            domain_upper = domain.upper()
            xpt_filename = f"{domain.lower()}.xpt"
            sha256 = source_sha256s.get(domain, "")

            # ItemGroupDef for the dataset
            igdef_attrib = {
                "OID": f"IG.{domain_upper}",
                "Name": domain_upper,
                "Repeating": "Yes",
                "IsReferenceData": "Yes",
                f"{{{_DEF_NS}}}Structure": "One record per parameter",
                f"{{{_DEF_NS}}}DomainDescription": domain_upper,
                f"{{{_DEF_NS}}}ArchiveLocationID": f"LF.{domain_upper}",
            }
            igdef = ET.SubElement(
                mdv, f"{{{_ODM_NS}}}ItemGroupDef", attrib=igdef_attrib
            )
            desc = ET.SubElement(igdef, f"{{{_ODM_NS}}}Description")
            ET.SubElement(desc, f"{{{_ODM_NS}}}TranslatedText").text = (
                f"SDTM Trial Design {domain_upper}"
                + (f" [source sha256: {sha256[:16]}...]" if sha256 else "")
            )

            # One ItemRef per column
            for col in df.columns:
                item_oid = f"IT.{domain_upper}.{col}"
                ET.SubElement(
                    igdef,
                    f"{{{_ODM_NS}}}ItemRef",
                    attrib={
                        "ItemOID": item_oid,
                        "Mandatory": "No",
                        "OrderNumber": str(list(df.columns).index(col) + 1),
                    },
                )

            # ItemDefs for each unique variable
            var_meta = _VAR_ATTRS.get(domain_upper, {})
            for col in df.columns:
                item_oid = f"IT.{domain_upper}.{col}"
                if item_oid in itemdef_added:
                    continue
                itemdef_added.add(item_oid)
                attrs = var_meta.get(col, {})
                datatype = "float" if attrs.get("type") == "Num" else "text"
                idef = ET.SubElement(
                    mdv,
                    f"{{{_ODM_NS}}}ItemDef",
                    attrib={
                        "OID": item_oid,
                        "Name": col,
                        "DataType": datatype,
                        "Length": attrs.get("length", "200"),
                        f"{{{_DEF_NS}}}Label": attrs.get("label", col),
                    },
                )
                desc2 = ET.SubElement(idef, f"{{{_ODM_NS}}}Description")
                ET.SubElement(desc2, f"{{{_ODM_NS}}}TranslatedText").text = (
                    attrs.get("label", col)
                )

            # Leaf / archive file location (href is a plain attribute here;
            # the namespace context is already declared via nsmap)
            ET.SubElement(
                mdv,
                f"{{{_DEF_NS}}}leaf",
                attrib={
                    "ID": f"LF.{domain_upper}",
                    "href": xpt_filename,
                },
            )

        # Serialise
        if _LXML:
            return ET.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=True)
        else:
            ET.indent(root)  # type: ignore[attr-defined]
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
