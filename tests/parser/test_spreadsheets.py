"""Tests for structural XLSX row extraction."""

from __future__ import annotations

import io
import zipfile

from legal_rag.parser.spreadsheets import extract_xlsx_rows


def _xlsx_bytes() -> bytes:
    """Build one minimal workbook with shared, numeric, and formula cells."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets><sheet name="Theo doi" sheetId="1" r:id="rId1"/></sheets>
            </workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
            </Relationships>""",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            """<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>Bien ban</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1">
                  <c r="A1" t="s"><v>0</v></c>
                  <c r="C1"><f>1+1</f><v>2</v></c>
                </row>
              </sheetData>
            </worksheet>""",
        )
    return buffer.getvalue()


def test_extract_xlsx_rows_preserves_sheet_cells_and_formulas() -> None:
    """Workbook order and native cell references remain searchable."""
    rows = extract_xlsx_rows(_xlsx_bytes(), "forms.xlsx")

    assert len(rows) == 1
    assert rows[0].sheet_name == "Theo doi"
    assert rows[0].source_ref == "xl/worksheets/sheet1.xml#/row/1"
    assert rows[0].text == "Bien ban\t2"
    assert [cell.to_dict() for cell in rows[0].cells] == [
        {"cell_ref": "A1", "value": "Bien ban", "formula": None},
        {"cell_ref": "C1", "value": "2", "formula": "1+1"},
    ]
