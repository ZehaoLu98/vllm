#!/usr/bin/env python3
"""
Build a forest of compressed tries (radix trees) from prompts.json
and generate an interactive HTML visualization.

Each tree in the forest corresponds to one system_prompt value.
Within each tree, the radix tree is built on (descriptive_text + query).
Traversing from root to leaf and concatenating node labels yields the full prompt text.

Usage:
    python build_forest.py [--input prompts.json] [--output forest.html]
"""

import argparse
import html as html_mod
import json
from pathlib import Path


class RadixNode:
    """A node in a compressed trie (radix tree)."""

    __slots__ = ("label", "children", "prompt_data")

    def __init__(self, label=""):
        self.label = label
        self.children: dict[str, "RadixNode"] = {}
        self.prompt_data: list[tuple[int, int]] = []

    def insert(self, string: str, prompt_idx: int, desc_len: int) -> None:
        if not string:
            self.prompt_data.append((prompt_idx, desc_len))
            return

        fc = string[0]
        if fc not in self.children:
            node = RadixNode(string)
            node.prompt_data.append((prompt_idx, desc_len))
            self.children[fc] = node
            return

        child = self.children[fc]
        common = 0
        limit = min(len(child.label), len(string))
        while common < limit and child.label[common] == string[common]:
            common += 1

        if common == len(child.label):
            child.insert(string[common:], prompt_idx, desc_len)
        elif common == len(string):
            split = RadixNode(string)
            split.prompt_data.append((prompt_idx, desc_len))
            child.label = child.label[common:]
            split.children[child.label[0]] = child
            self.children[fc] = split
        else:
            split = RadixNode(child.label[:common])
            child.label = child.label[common:]
            split.children[child.label[0]] = child
            new_node = RadixNode(string[common:])
            new_node.prompt_data.append((prompt_idx, desc_len))
            split.children[string[common]] = new_node
            self.children[fc] = split

    def count_prompts(self) -> int:
        n = len(self.prompt_data)
        for c in self.children.values():
            n += c.count_prompts()
        return n

    def _all_desc_lens(self) -> list[int]:
        lens = [dl for _, dl in self.prompt_data]
        for c in self.children.values():
            lens.extend(c._all_desc_lens())
        return lens

    def to_dict(self) -> dict:
        d: dict = {"label": self.label, "count": self.count_prompts()}
        all_dl = self._all_desc_lens()
        if all_dl:
            d["min_desc_len"] = min(all_dl)
        if self.prompt_data:
            d["prompts"] = sorted(idx for idx, _ in self.prompt_data)
        if self.children:
            d["children"] = [
                c.to_dict()
                for c in sorted(self.children.values(), key=lambda n: n.label)
            ]
        return d


def build_forest(prompts: list[dict]) -> list[dict]:
    groups: dict[str, list[tuple[int, int, str]]] = {}
    for i, p in enumerate(prompts):
        sp = p["system_prompt"]
        groups.setdefault(sp, []).append(
            (i, len(p["descriptive_text"]), p["descriptive_text"] + p["query"])
        )

    forest = []
    for sp in sorted(groups):
        root = RadixNode("")
        for idx, desc_len, text in groups[sp]:
            root.insert(text, idx, desc_len)
        tree = {
            "label": sp,
            "count": root.count_prompts(),
            "children": [
                c.to_dict()
                for c in sorted(root.children.values(), key=lambda n: n.label)
            ],
        }
        forest.append(tree)
    return forest


# --------------- HTML generation ---------------

def _esc(text: str) -> str:
    return html_mod.escape(text, quote=True)


def _trunc(text: str, maxlen: int = 80) -> str:
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 1] + "\u2026"


def generate_html(forest: list[dict], prompts: list[dict]) -> str:
    total = sum(t["count"] for t in forest)
    # Compute some stats
    branch_nodes = 0
    leaf_nodes = 0
    max_depth = 0

    def _walk(node, depth):
        nonlocal branch_nodes, leaf_nodes, max_depth
        is_leaf = "children" not in node
        if is_leaf:
            leaf_nodes += 1
            max_depth = max(max_depth, depth)
        else:
            if len(node.get("children", [])) > 1 or node.get("prompts"):
                branch_nodes += 1
            for ch in node.get("children", []):
                _walk(ch, depth + 1)

    for tree in forest:
        for ch in tree.get("children", []):
            _walk(ch, 1)

    # Render the tree nodes as HTML
    def render_node(node, depth, offset=0):
        label = node["label"]
        count = node["count"]
        has_children = "children" in node
        plist = node.get("prompts", [])
        min_dl = node.get("min_desc_len", 0)
        full = _esc(label)
        tag_html = "".join(
            f'<span class="ptag">#{p}</span>' for p in plist
        )

        # Color label by content type: descriptive_text vs query
        label_end = offset + len(label)
        if label_end <= min_dl:
            lbl_html = f'<span class="lbl lbl-desc" title="{full}">{_esc(_trunc(label, 80))}</span>'
        elif offset >= min_dl:
            lbl_html = f'<span class="lbl lbl-query" title="{full}">{_esc(_trunc(label, 80))}</span>'
        else:
            sp = min_dl - offset
            lbl_html = (
                f'<span class="lbl lbl-desc" title="{full}">{_esc(_trunc(label[:sp], 40))}</span>'
                f'<span class="lbl lbl-query">{_esc(_trunc(label[sp:], 40))}</span>'
            )

        if has_children:
            n_children = len(node["children"])
            cls = "branch" if n_children > 1 else "chain"
            inner = "\n".join(render_node(ch, depth + 1, offset + len(label)) for ch in node["children"])
            return (
                f'<div class="nd {cls}" data-depth="{depth}">'
                f'<div class="hdr" onclick="tog(this)">'
                f'<span class="arrow">&#9654;</span>'
                f'{lbl_html}'
                f'{tag_html}'
                f'<span class="cnt">{count}</span>'
                f'</div>'
                f'<div class="ch">{inner}</div>'
                f'</div>'
            )
        else:
            return (
                f'<div class="nd leaf" data-depth="{depth}">'
                f'<span class="dot">\u25CF</span>'
                f'{lbl_html}'
                f'{tag_html}'
                f'</div>'
            )

    trees_html = ""
    for tree in forest:
        sp_esc = _esc(tree["label"])
        kids = "\n".join(render_node(ch, 1) for ch in tree["children"])
        trees_html += (
            f'<div class="tree">'
            f'<div class="root-hdr" onclick="togRoot(this)">'
            f'<span class="arrow">&#9660;</span>'
            f'<span class="root-label">{sp_esc}</span>'
            f'<span class="cnt">{tree["count"]}</span>'
            f'</div>'
            f'<div class="root-ch open">{kids}</div>'
            f'</div>\n'
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Prompt Forest Visualization</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;line-height:1.5}}
h1{{color:#58a6ff;margin-bottom:4px;font-size:1.6rem}}
.subtitle{{color:#8b949e;margin-bottom:20px;font-size:.95rem}}
.stats{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 18px;min-width:140px}}
.stat .val{{font-size:1.5rem;font-weight:700;color:#58a6ff}}
.stat .lbl2{{font-size:.8rem;color:#8b949e;margin-top:2px}}
.controls{{margin-bottom:18px;display:flex;gap:10px;align-items:center}}
.controls button{{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.85rem}}
.controls button:hover{{background:#30363d}}
.controls input{{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 12px;border-radius:6px;width:260px;font-size:.85rem}}
.tree{{margin-bottom:28px}}
.root-hdr{{background:#1c2333;padding:10px 14px;border-radius:8px 8px 0 0;border:1px solid #30363d;cursor:pointer;display:flex;align-items:center;gap:8px;user-select:none}}
.root-hdr:hover{{background:#222d3d}}
.root-label{{font-weight:700;color:#f0883e;font-size:1.05rem}}
.root-ch{{border:1px solid #30363d;border-top:none;border-radius:0 0 8px 8px;padding:4px 0;max-height:70vh;overflow-y:auto;display:none}}
.root-ch.open{{display:block}}
.nd{{padding:1px 0 1px 20px;position:relative}}
.nd::before{{content:'';position:absolute;left:9px;top:0;bottom:0;width:1px;background:#30363d}}
.hdr{{display:flex;align-items:center;gap:6px;cursor:pointer;padding:2px 6px;border-radius:4px;user-select:none}}
.hdr:hover{{background:#1c2333}}
.arrow{{font-size:.65rem;color:#484f58;width:12px;display:inline-block;transition:transform .15s;flex-shrink:0}}
.nd.open>.hdr>.arrow{{transform:rotate(90deg)}}
.ch{{display:none;padding-left:4px}}
.nd.open>.ch{{display:block}}
.dot{{color:#3fb950;font-size:.6rem;margin-right:4px}}
.lbl{{font-family:'Cascadia Code','Fira Code',monospace;font-size:.78rem;word-break:break-all}}
.lbl-desc{{color:#3fb950}}
.lbl-query{{color:#d2a8ff}}
.cnt{{font-size:.7rem;color:#484f58;margin-left:auto;padding-right:10px;flex-shrink:0;white-space:nowrap}}
.ptag{{font-size:.65rem;background:#1f6feb33;color:#58a6ff;border-radius:3px;padding:0 5px;margin-left:2px;white-space:nowrap}}
.legend{{margin-bottom:18px;display:flex;gap:16px;flex-wrap:wrap;font-size:.82rem}}
.legend span{{display:flex;align-items:center;gap:5px}}
.legend .sw{{width:12px;height:12px;border-radius:3px;display:inline-block}}
.match-hl{{background:#f0883e33 !important}}
</style>
</head>
<body>
<h1>Prompt Forest &mdash; Radix Tree Visualization</h1>
<p class="subtitle">Each path from root to leaf concatenates to: <b>system_prompt</b> + <b>descriptive_text</b> + <b>query</b>.
Nodes with multiple children indicate prefix sharing between prompts.</p>

<div class="stats">
  <div class="stat"><div class="val">{total}</div><div class="lbl2">Total prompts</div></div>
  <div class="stat"><div class="val">{len(forest)}</div><div class="lbl2">System prompts (trees)</div></div>
  <div class="stat"><div class="val">{branch_nodes}</div><div class="lbl2">Branch nodes (prefix sharing)</div></div>
  <div class="stat"><div class="val">{leaf_nodes}</div><div class="lbl2">Leaf nodes</div></div>
  <div class="stat"><div class="val">{max_depth}</div><div class="lbl2">Max tree depth</div></div>
</div>

<div class="legend">
  <span><span class="sw" style="background:#f0883e"></span> System prompt (root)</span>
  <span><span class="sw" style="background:#3fb950"></span> Descriptive text</span>
  <span><span class="sw" style="background:#d2a8ff"></span> Query</span>
  <span><span class="sw" style="background:#58a6ff"></span> Prompt index tag</span>
</div>

<div class="controls">
  <button onclick="expandAll()">Expand all</button>
  <button onclick="collapseAll()">Collapse all</button>
  <button onclick="expandBranches()">Expand branches only</button>
  <input type="text" id="search" placeholder="Search prompt # or text\u2026" oninput="doSearch(this.value)">
</div>

<div id="forest">
{trees_html}
</div>

<script>
function tog(hdr){{
  hdr.parentElement.classList.toggle('open');
}}
function togRoot(hdr){{
  hdr.nextElementSibling.classList.toggle('open');
  const arrow=hdr.querySelector('.arrow');
  arrow.innerHTML=hdr.nextElementSibling.classList.contains('open')?'&#9660;':'&#9654;';
}}
function expandAll(){{
  document.querySelectorAll('.nd').forEach(n=>n.classList.add('open'));
  document.querySelectorAll('.root-ch').forEach(n=>n.classList.add('open'));
  document.querySelectorAll('.root-hdr .arrow').forEach(a=>a.innerHTML='&#9660;');
}}
function collapseAll(){{
  document.querySelectorAll('.nd').forEach(n=>n.classList.remove('open'));
  document.querySelectorAll('.root-ch').forEach(n=>{{n.classList.remove('open');}});
  document.querySelectorAll('.root-hdr .arrow').forEach(a=>a.innerHTML='&#9654;');
}}
function expandBranches(){{
  collapseAll();
  document.querySelectorAll('.root-ch').forEach(n=>n.classList.add('open'));
  document.querySelectorAll('.root-hdr .arrow').forEach(a=>a.innerHTML='&#9660;');
  document.querySelectorAll('.nd.branch').forEach(n=>n.classList.add('open'));
}}
function doSearch(q){{
  document.querySelectorAll('.match-hl').forEach(n=>n.classList.remove('match-hl'));
  if(!q) return;
  const low=q.toLowerCase();
  document.querySelectorAll('.ptag').forEach(tag=>{{
    if(tag.textContent.toLowerCase().includes(low)){{
      tag.classList.add('match-hl');
      let el=tag.parentElement;
      while(el){{
        if(el.classList.contains('nd')) el.classList.add('open');
        if(el.classList.contains('root-ch')) el.classList.add('open');
        el=el.parentElement;
      }}
    }}
  }});
  document.querySelectorAll('.lbl').forEach(lbl=>{{
    const full=lbl.getAttribute('title')||lbl.textContent;
    if(full.toLowerCase().includes(low)){{
      lbl.classList.add('match-hl');
      let el=lbl.parentElement;
      while(el){{
        if(el.classList.contains('nd')) el.classList.add('open');
        if(el.classList.contains('root-ch')) el.classList.add('open');
        el=el.parentElement;
      }}
    }}
  }});
}}
// Start with branches expanded
expandBranches();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build prompt forest visualization")
    parser.add_argument("--input", default="prompts.json", help="Input prompts JSON file")
    parser.add_argument("--output", default="forest.html", help="Output HTML file")
    args = parser.parse_args()

    with open(args.input) as f:
        prompts = json.load(f)

    forest = build_forest(prompts)

    html = generate_html(forest, prompts)
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Wrote forest visualization ({len(prompts)} prompts) to {args.output}")

    # Also write the JSON structure for reference
    json_out = Path(args.output).with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(forest, f, indent=2)
    print(f"Wrote forest JSON to {json_out}")


if __name__ == "__main__":
    main()
