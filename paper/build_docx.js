// Build manuscript.docx from manuscript.md with faithful formatting.
// Handles: #/##/### headings, paragraphs with **bold** and *italic*,
// pipe tables (bordered, bold header), embedded PNG figures scaled to text
// width, italic figure captions, and the reference list.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, WidthType, BorderStyle, TableLayoutType,
} = require("docx");

const MD = path.join(__dirname, "manuscript.md");
const OUT = path.join(__dirname, "manuscript.docx");
const FIGDIR = __dirname;

const IMG_DIMS = {
  "f1_dataset.png": [2200, 640],
  "f2_spectrograms.png": [2400, 560],
  "f3_confusions.png": [1920, 780],
  "f4_robustness.png": [920, 640],
  "f5_attention.png": [1800, 800],
  "f6_umap.png": [2000, 840],
};

const TEXT_WIDTH_PX = 624; // 6.5 inches at 96 dpi (US Letter, 1 inch margins)
const TEXT_WIDTH_DXA = 9360;
const FONT = "Times New Roman";
const SIZE = 24; // half-points = 12pt

// Parse **bold** and *italic* into TextRuns. No nesting in this document.
function runs(text, extra = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SIZE, ...extra }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, size: SIZE, ...extra }));
    } else {
      out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, size: SIZE, ...extra }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SIZE, ...extra }));
  return out;
}

function para(text, opts = {}) {
  return new Paragraph({
    children: runs(text, opts.runExtra || {}),
    spacing: { after: 160, line: 276 },
    alignment: AlignmentType.LEFT,
    ...opts.paraExtra,
  });
}

const cellBorders = {
  top: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
  bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
  left: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
  right: { style: BorderStyle.SINGLE, size: 4, color: "000000" },
};

function buildTable(headerCells, rows) {
  // first column gets a double share so model and species names sit on one line
  const nCols = headerCells.length;
  const colW = Math.floor(TEXT_WIDTH_DXA / (nCols + 1));
  const widths = Array(nCols).fill(colW);
  widths[0] = TEXT_WIDTH_DXA - colW * (nCols - 1);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headerCells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      borders: cellBorders,
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(c, { bold: true }), spacing: { after: 0 } })],
    })),
  });
  const bodyRows = rows.map(cells => new TableRow({
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      borders: cellBorders,
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(c), spacing: { after: 0 } })],
    })),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: TEXT_WIDTH_DXA, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    rows: [headerRow, ...bodyRows],
  });
}

function splitTableLine(line) {
  return line.replace(/^\|/, "").replace(/\|$/, "").split("|").map(s => s.trim());
}

const lines = fs.readFileSync(MD, "utf8").split("\n");
const children = [];
let i = 0;
while (i < lines.length) {
  const line = lines[i];

  if (line.trim() === "" || line.trim() === "---") { i++; continue; }

  if (line.startsWith("# ")) {
    children.push(new Paragraph({
      children: runs(line.slice(2), { bold: true, size: 32 }),
      heading: HeadingLevel.TITLE,
      spacing: { after: 240 },
    }));
    i++; continue;
  }
  if (line.startsWith("## ")) {
    children.push(new Paragraph({
      children: runs(line.slice(3), { bold: true, size: 28 }),
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 280, after: 160 },
    }));
    i++; continue;
  }
  if (line.startsWith("### ")) {
    children.push(new Paragraph({
      children: runs(line.slice(4), { bold: true, italics: true, size: 24 }),
      heading: HeadingLevel.HEADING_2,
      spacing: { before: 240, after: 120 },
    }));
    i++; continue;
  }

  // image
  const img = line.match(/^!\[[^\]]*\]\(([^)]+)\)/);
  if (img) {
    const rel = img[1];
    const base = path.basename(rel);
    const [w, h] = IMG_DIMS[base] || [1600, 800];
    const scale = Math.min(1, TEXT_WIDTH_PX / w);
    children.push(new Paragraph({
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(path.join(FIGDIR, rel)),
        transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
      })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 80 },
    }));
    i++; continue;
  }

  // table
  if (line.startsWith("|") && i + 1 < lines.length && /^\|[\s:-]+\|/.test(lines[i + 1].replace(/-/g, "-"))) {
    const header = splitTableLine(line);
    let j = i + 2;
    const rows = [];
    while (j < lines.length && lines[j].startsWith("|")) {
      rows.push(splitTableLine(lines[j]));
      j++;
    }
    children.push(buildTable(header, rows));
    children.push(new Paragraph({ children: [], spacing: { after: 120 } }));
    i = j; continue;
  }

  // figure caption (italic line starting with *Figure)
  if (/^\*Figure \d/.test(line.trim())) {
    children.push(new Paragraph({
      children: runs(line.trim()),
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }));
    i++; continue;
  }

  // normal paragraph
  children.push(para(line));
  i++;
}

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: SIZE } },
      title: { run: { font: FONT, size: 32, bold: true, color: "000000" } },
      heading1: { run: { font: FONT, size: 28, bold: true, color: "000000" } },
      heading2: { run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, buf.length, "bytes");
});
