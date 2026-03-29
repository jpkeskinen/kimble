# Kimble – Python-toteutus

Python-toteutus suomalaisesta **Kimble**-lautapelistä (tunnetaan myös nimellä Trouble). Tukee ihmis- ja tietokonepelaajia, massasimulaatiota sekä neuroverkkoihin perustuvaa tekoälyä.

---

## Pelin säännöt lyhyesti

- 4 pelaajaa, jokaisella 4 nappulaa kotipesässä
- 28 ruutua pääradalla, 4 ruutua maalialueella
- **Kuutosella** pääsee kotipesästä radalle; kuutonen antaa myös lisäheiton
- Laskeutuminen vastustajan nappulan päälle lähettää sen takaisin kotipesään (*torkku*)
- Kolme kuutosta peräkkäin katkaisee vuoron
- Voittaja: ensimmäinen joka saa kaikki 4 nappulaa maaliin

---

## Tiedostorakenne

| Tiedosto | Kuvaus |
|---|---|
| `board.py` | Laudan vakiot (`TRACK_SIZE=28`, `HOME_SIZE=4`) ja `Piece`-luokka |
| `game.py` | Pelilogiikka, siirtosäännöt, torkkaaminen, ASCII-näyttö |
| `player.py` | `Player`-luokka ja kaikki 9 tekoälystrategiaa |
| `nn_player.py` | Neuroverkkoluokat ja tilavektori |
| `train_nn.py` | Harjoittelukoodi (REINFORCE ja Actor-Critic) |
| `main.py` | Käyttöliittymä: interaktiivinen peli ja simulaatiomoodi |

---

## Käyttö

```bash
# Aktivoi virtuaaliympäristö
source venv/bin/activate

# Interaktiivinen peli (valitse ihmis- tai tietokonepelaajat)
python main.py

# Harjoittele matala neuroverkko (REINFORCE, ~40 min GPU)
python train_nn.py --episodes 100000

# Harjoittele syvä neuroverkko
python train_nn.py --episodes 100000 --deep

# Harjoittele Actor-Critic + self-play (~4 h GPU)
python train_nn.py --episodes 50000 --ac --lr 3e-4

# Muuta batch-kokoa tai laitetta
python train_nn.py --episodes 100000 --batch-size 64 --device cuda
```

---

## Tekoälystrategiat

| Avain | Nimi | Kuvaus |
|---|---|---|
| `random` | Satunnainen | Valitsee liikutettavan nappulan satunnaisesti |
| `longest` | Pisin ensin | Siirtää pisimmälle edennyttä nappulaa; kuutosella uusi nappula kotipesästä |
| `longest_eat` | Pisin ensin + syö | Kuten `longest`, mutta syö vastustajan aina kun mahdollista |
| `longest_eat_dodge` | Pisin ensin + syö + väistä | Kuten `longest_eat`, lisäksi väistää uhattuja nappuloita |
| `shortest` | Lyhin ensin | Siirtää lähimpänä lähtöä olevaa nappulaa |
| `shortest_eat` | Lyhin ensin + syö | Kuten `shortest`, mutta syö vastustajan aina kun mahdollista |
| `nn` | Neuroverkko (matala) | REINFORCE-politiikkagradientti, KimbleNet |
| `nn_deep` | Neuroverkko (syvä) | REINFORCE-politiikkagradientti, KimbleDeepNet |
| `nn_ac` | Neuroverkko (Actor-Critic) | Actor-Critic + self-play, KimbleActorCritic |

---

## Neuroverkkojen arkkitehtuuri

### Tilavektori (34 piirrettä)

| Piirteet | Kuvaus |
|---|---|
| 4 | Omat nappulat: normalisoitu eteneminen (0–1) |
| 12 | Vastustajien nappulat: normalisoitu eteneminen (3 pelaajaa × 4) |
| 6 | Nopan silmäluku: one-hot-koodaus |
| 4 | Liikutettavuusmaski |
| 4 | Syömismahdollisuusmaski |
| 4 | Uhkamaski (nappulat jotka vastustaja voi syödä seuraavalla siirrolla) |

### Verkkoarkkitehtuurit

```
KimbleNet         34 → 128 → 64 → 4
KimbleDeepNet     34 → 256 → 256 → 128 → 64 → 4
KimbleActorCritic 34 → 256 → 128 → actor: 4  (politiikka)
                                 → critic: 1  (arvoarvio V(s))
```

### Harjoittelumenetelmät

**REINFORCE** (`nn`, `nn_deep`): Politiikkagradientti diskontoiduilla palautuksilla (γ=0,99) ja advantage-normalisoinnilla. Vain pelaaja 0 harjoittelee per episodi.

**Actor-Critic + self-play** (`nn_ac`): Kaikki 4 pelaajaa käyttävät samaa mallia. 70 % peleistä täysi self-play, 30 % pelaaja 0 vastaan historiallisia malleja (opponent pool, 10 viimeistä). Loss = actor + 0,5 × critic − 0,01 × entropia.

**Reward shaping** (kaikki NN-mallit):
- Nappulan eteneminen: +0,005 per normalisoitu yksikkö
- Vastustajan syöminen: +0,3
- Syödyksi tuleminen: −0,2
- Nappula viimeiseen maalislottiin: +0,5

---

## Tulokset

### Strategioiden vertailu (round-robin, n=10 000)

| Strategia | Voitto-% |
|---|---|
| Pisin ensin + syö | **23,1 %** |
| Pisin ensin + syö + väistä | 22,4 % |
| Pisin ensin | 19,5 % |
| Satunnainen | 17,8 % |
| Lyhin ensin + syö | 10,2 % |
| Lyhin ensin | 6,9 % |

### NN-vertailu tasapainoisessa pelissä (n=25 000)

Kokoonpano: Neuroverkko (AC) + Satunnainen + Pisin ensin + syö + Neuroverkko (matala)

| Strategia | Voitto-% |
|---|---|
| Pisin ensin + syö | 33,0 % |
| Neuroverkko (matala) | 23,9 % |
| Neuroverkko (AC) | 23,0 % |
| Satunnainen | 20,1 % |

---

## Miksi neuroverkot eivät voita parasta sääntöstrategiaa?

Lyhyt vastaus: **paras sääntöpohjainen strategia on jo lähellä optimaalista tässä pelissä**.

Pidempi selitys:

1. **Noppa dominoi.** Joka vuorolla heitetään noppaa — iso osa pelin lopputuloksesta on sattumaa. Gradient-signaali hukkuu kohinaan, kun yksittäisen siirron vaikutus voittoon näkyy vasta satoja siirtoja myöhemmin.

2. **Päätöksenteko on yksinkertaista.** Tyypillisesti liikutettavia nappuloita on 1–3. Pelissä ei ole monimutkaisia kombinatorisia valintoja joita neuroverkon täytyisi oppia.

3. **`longest_eat` käyttää jo kaiken oleellisen informaation.** Se syö aina kun voi (optimaalinen) ja etenee pisimmällä nappulalla (hyvä heuristiikka). Neuroverkolle ei jää juuri mitään lisäopittavaa.

4. **`longest_eat_dodge` osoittaa väistämisen olevan haitallista.** Uhattuna olevan nappulan väistäminen heikentää tulosta, koska syöjä tarvitsee tietyn nopanluvun — odotusarvo syödyksi tulemiselle on matala. Neuroverkon oppima "parempi" strategia kohtaa saman ongelman.

Realistinen yläraja neuroverkolle tässä pelissä on noin 25–27 % round-robinissa. Merkittävä parannus vaatisi joko paljon rikkaampaa tilaesitystä tai hakupohjaista lähestymistapaa (esim. Monte Carlo Tree Search nopanheiton odotusarvolaskennalla).

---

## Riippuvuudet

```
torch       (PyTorch, neuroverkot)
tqdm        (edistymispalkki simulaatiossa ja harjoittelussa)
```

Asenna virtuaaliympäristöön:

```bash
python -m venv venv
source venv/bin/activate
pip install torch tqdm
```
