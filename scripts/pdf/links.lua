-- Pandoc filter for scripts/build_pdfs.sh: relative links (../reports/benchmark.md) resolve against the
-- source file's folder and point at that file on GitHub, so they still work once the doc is a PDF.
local REPO = "https://github.com/Ashish94312/floor-plan/blob/main/"

function Link(el)
  local t = el.target
  if t:match("^%a[%w+.-]*:") or t:sub(1, 1) == "#" then
    return el
  end
  local dir = PANDOC_STATE.input_files[1]:match("^(.*)/") or "."
  local parts = {}
  for p in (dir .. "/" .. t):gmatch("[^/]+") do
    if p == ".." then
      table.remove(parts)
    elseif p ~= "." then
      table.insert(parts, p)
    end
  end
  el.target = REPO .. table.concat(parts, "/")
  return el
end
