# FoodScan – saját tárhelyre (élő kamera + irányítás)

A Claude-ban megnyitott oldal biztonsági okból nem kérhet kamerát, ezért az élő, irányított felvételhez
ezt a 4 fájlt egy saját HTTPS-oldalra kell tenni. A legegyszerűbb a GitHub Pages:

1. github.com → New repository (pl. `foodscan`), Public.
2. „Add file → Upload files”: húzd be az `index.html`, `core.js`, `analyze.js`, `foods.js` fájlokat → Commit.
3. Settings → Pages → Source: „Deploy from a branch”, Branch: `main` / `(root)` → Save.
4. Kb. 1 perc múlva elérhető: `https://<felhasználónév>.github.io/foodscan/` — ezt nyisd meg Safariban,
   és engedélyezd a kamerát.

Felvétel: „Élő kamera” → a narancs nyíl mutatja, merre mozdítsd a telefont, a jobb alsó kis kupola a
lefedettséget (zöld = megvan, karika = a következő cél). Ha a sáv zöldre vált („Kész ✓”), állítsd le,
majd „Export (ZIP)”.

Elemzés: a Claude-os ételfelismerés csak a Claude-ban megnyitott változatban működik — ott a
„ZIP betöltése” gombbal töltsd be az exportot, és nyomd meg az „Automatikus elemzés” gombot.
(Itt is működik az elemzés, csak az ételt kézzel kell kiválasztani.)
