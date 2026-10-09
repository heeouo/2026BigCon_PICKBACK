"""
최종 평소 매출 baseline (V4) — 스크립트 버전
baseline_v4.ipynb 와 같은 코드입니다. Colab이 아닌 환경에서 실행할 때 사용하세요.

실행:
    pip install -r requirements.txt
    python baseline_v4.py --data_dir ./data --out_dir ./outputs        # GPU 있으면 자동 사용
    python baseline_v4.py --data_dir ./data --out_dir ./outputs --cpu  # 결과를 완전히 같게 재현하려면 CPU

출력: {out_dir}/38_최종baseline_V4.csv, {out_dir}/38_최종baseline_V4_검증요약.csv
"""
import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--data_dir', default='./data', help='원본 파일 3개가 있는 폴더 (하위 폴더까지 찾음)')
_p.add_argument('--out_dir', default='./outputs')
_p.add_argument('--cpu', action='store_true', help='GPU가 있어도 CPU로 학습')
ARGS = _p.parse_args()

def display(x):
    print(x.to_string() if hasattr(x, 'to_string') else x)

# ======================================================================
# 0. 준비: 패키지, 경로, 설정
# ======================================================================
import os, re, glob, time, warnings, subprocess, sys
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
try: import xgboost
except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost'])
import xgboost as xgb
print('xgboost', xgb.__version__, '/ pandas', pd.__version__, '/ numpy', np.__version__)

# ---- 경로 ----
DATA_DIR = ARGS.data_dir
OUT_DIR = ARGS.out_dir
os.makedirs(OUT_DIR, exist_ok=True)
def find(pattern):
    hits = sorted(glob.glob(f'{DATA_DIR}/**/{pattern}', recursive=True))
    if not hits: raise FileNotFoundError(f'{DATA_DIR}에서 {pattern}을 찾지 못했습니다')
    return hits[0]
CARD_PATH = find('신한카드_빅콘테스트2026_데이터1*')
WEATHER_PATH = find('강남_춘천_기상데이터_정제*.csv')
try: FLAG_PATH = find('특보데이터_강남춘천_일별발효_등급*.csv')
except FileNotFoundError: FLAG_PATH = find('특보데이터_강남춘천_일별발효*.csv')
print(CARD_PATH, WEATHER_PATH, FLAG_PATH, sep='\n')

# ---- 전처리 설정 ----
REGION_MAP = {'서울 강남구': '강남(서울동남권)', '강원 춘천시': '춘천'}
EXCLUDE_RY = ['세금공과금', 'ZZ_나머지']     # 법인(SEX_CCD)도 제외
MIN_NONZERO = 0.8      # 칸(지역×업종×시간대)의 매출>0 날 비율이 이 값 미만이면 희소 칸으로 제외
RAIN_MAX = 10          # 평시 = 폭염·호우 특보 없음 & 일강수 < 10mm
CLOSED_SHARE = 0.5     # 휴무 칸 = 그 칸·그 요일(공휴일은 따로)의 평시 날 중 매출 0 비율 ≥ 0.5

# ---- 달력 설정 ----
HOLI = pd.to_datetime(['2025-08-15', '2025-10-03', '2025-10-05', '2025-10-06', '2025-10-07', '2025-10-08',
                       '2025-10-09', '2025-12-25'])          # 2025 하반기 공휴일(대체공휴일 포함)
CHUSEOK_PRE = ('2025-09-22', '2025-10-02')
YEAR_END = ('2025-12-22', '2025-12-31')
XMAS_EVE = pd.Timestamp('2025-12-24')
SPECIAL_DAYS = pd.to_datetime(['2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',
                               '2025-10-08', '2025-10-09', '2025-12-24', '2025-12-25'])   # 검증에서 따로 볼 특수일(학습엔 그대로 씀)

# ---- 교차 적합 ----
N_FOLDS, BUFFER = 5, 3

# ---- 주기 달력을 넣는 업종: (지역 앞글자, 업종명에 포함된 글자) ----
CYCLE_INDS = [('강남', '유치원'), ('춘천', '유치원')]

# ---- V4 공휴일 → 일요일 처리 업종 ----
# 강남: 전 업종 / 춘천: 공휴일·일요일 유사성 표에서 (모든 시간대 유사 & 가중차이 ≤10%p) 또는 (가중차이 ≤5%p)인 업종
HOLI_SUN_ALL_REGIONS = ['강남']
HOLI_SUN_LIST = [('춘천', u) for u in ['LPG가스', '동물병원', '약국', '일반병원', '자동차용품', '종합병원', '주유소',
                                     '중고품판매점', '컴퓨터/소프트웨어', '학원/학습지', '화장품']]

# ---- 모델 ----
USE_GPU = not ARGS.cpu
try: gpu = USE_GPU and subprocess.run('nvidia-smi', capture_output=True).returncode == 0
except FileNotFoundError: gpu = False
XGB_PARAMS = dict(objective='reg:tweedie', tweedie_variance_power=1.5,
                  n_estimators=1000, learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=1.0,
                  enable_categorical=True, max_cat_to_onehot=1, tree_method='hist', random_state=42)
if gpu: XGB_PARAMS['device'] = 'cuda'
SCALE = 1e4            # 목표값 = 매출 / 1만 원
print('GPU 사용:', gpu)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)

# ======================================================================
# 1. 카드 매출 집계 → 빈 칸 0으로 채운 격자 → 희소 칸 제외
# ======================================================================
card = pd.read_csv(CARD_PATH, sep='\t', encoding='cp949', dtype={'TA_YMD': str, 'TIME_GB': str})
n0 = len(card)
card['TS_AT'] = pd.to_numeric(card['TS_AT'], errors='coerce')
card['USE_CNT'] = pd.to_numeric(card['USE_CNT'], errors='coerce')
card['날짜'] = pd.to_datetime(card['TA_YMD'], format='%Y%m%d')
card = card[(card['SEX_CCD'] != '법인') & (~card['MCT_RY_CD'].isin(EXCLUDE_RY))]
card['대상'] = card['MCT_SGG_CD'].map(REGION_MAP)
print(f'원본 {n0:,}행 → 법인·세금공과금·ZZ_나머지 제외 후 {len(card):,}행 / 지역 매핑 실패 {card["대상"].isna().sum():,}행')
agg = card.dropna(subset=['대상']).groupby(['날짜', '대상', 'MCT_RY_CD', 'TIME_GB'])[['TS_AT', 'USE_CNT']].sum()
del card

DATES = pd.date_range(agg.index.get_level_values(0).min(), agg.index.get_level_values(0).max())
grid = pd.MultiIndex.from_product([DATES, sorted(agg.index.get_level_values(1).unique()),
                                   sorted(agg.index.get_level_values(2).unique()), sorted(agg.index.get_level_values(3).unique())],
                                  names=['날짜', '대상', '업종', 'slot'])
P = agg.reindex(grid, fill_value=0).rename(columns={'TS_AT': 'S', 'USE_CNT': 'N'}).reset_index()
nz = P.groupby(['대상', '업종', 'slot'])['S'].apply(lambda s: (s > 0).mean())
keep = nz[nz >= MIN_NONZERO].index
P = P.set_index(['대상', '업종', 'slot']).loc[keep].reset_index()
SLOTS = sorted(P['slot'].unique(), key=lambda s: int(re.findall(r'\d+', s)[0]))
print(f'기간 {DATES.min().date()} ~ {DATES.max().date()} ({len(DATES)}일)')
print(f'남은 칸 {len(keep):,} / {len(nz):,}, 업종 수:', P.groupby('대상')['업종'].nunique().to_dict(), '/ 시간대:', SLOTS)

# ======================================================================
# 2. 특보 등급 붙이기, 한파·대설 날 제외
# ======================================================================
fl = pd.read_csv(FLAG_PATH, encoding='utf-8-sig', parse_dates=['날짜'])
warn_col = next((c for c in fl.columns if '경보' in c), None)
fl['등급'] = np.where(fl[warn_col].fillna(0) > 0, 2, np.where(fl['발효'] > 0, 1, 0)) if warn_col else (fl['발효'] > 0).astype(int)
G = fl.pivot_table(index=['대상', '날짜'], columns='카테고리', values='등급', aggfunc='max', fill_value=0).reset_index()
for c in ['폭염', '호우', '한파', '대설']:
    if c not in G: G[c] = 0
G = G.rename(columns={'폭염': '폭염등급', '호우': '호우등급'})
G['한파대설'] = (G['한파'] > 0) | (G['대설'] > 0)

P = P.merge(G[['대상', '날짜', '폭염등급', '호우등급', '한파대설']], on=['대상', '날짜'], how='left')
P[['폭염등급', '호우등급']] = P[['폭염등급', '호우등급']].fillna(0).astype(int)
cold = P['한파대설'].fillna(False).astype(bool)
print('한파·대설로 제외한 날:', P.loc[cold].groupby('대상')['날짜'].nunique().to_dict())
P = P[~cold].drop(columns='한파대설')
dd = P.drop_duplicates(['대상', '날짜'])
print('폭염일(주의보만/경보):', dd.groupby('대상')['폭염등급'].apply(lambda s: f'{(s == 1).sum()}/{(s == 2).sum()}').to_dict(),
      '/ 호우일:', dd.groupby('대상')['호우등급'].apply(lambda s: int((s > 0).sum())).to_dict())

# ======================================================================
# 3. 기상 붙이기
# ======================================================================
W = pd.read_csv(WEATHER_PATH, encoding='utf-8-sig', parse_dates=['일시'])
W['날짜'] = W['일시'].dt.normalize(); W['hour'] = W['일시'].dt.hour
starts = [int(re.findall(r'\d+', s)[0]) for s in SLOTS]
W['slot'] = W['hour'].map(lambda h: SLOTS[max(i for i, s in enumerate(starts) if s <= h)])
WD_ = W.groupby(['대상', '날짜']).agg(일최고기온=('기온(°C)', 'max'), 일강수=('강수량(mm)', 'sum')).reset_index()
WS_ = W.groupby(['대상', '날짜', 'slot']).agg(시간대최고기온=('기온(°C)', 'max'), 시간대강수=('강수량(mm)', 'sum'),
                                             시간대강수시간=('강수량(mm)', lambda s: int((s > 0).sum()))).reset_index()
P = P.merge(WD_, on=['대상', '날짜'], how='left').merge(WS_, on=['대상', '날짜', 'slot'], how='left')
print('기상 결측 비율:', P[['일최고기온', '일강수', '시간대최고기온']].isna().mean().round(4).to_dict())

# ======================================================================
# 4. 키·기본 달력·평시·교차적합 묶음
# ======================================================================
P['gkey'] = P['대상'] + '|' + P['업종']
P['skey'] = P['gkey'] + '#' + P['slot']
P['요일'] = P['날짜'].dt.dayofweek
P['주말'] = P['요일'].ge(5).astype(int)
P['공휴일'] = P['날짜'].isin(HOLI).astype(int)
P['daytype'] = np.where(P['공휴일'].eq(1), 7, P['요일'])
P['추석직전'] = P['날짜'].between(*CHUSEOK_PRE).astype(int)
P['평시'] = P['폭염등급'].eq(0) & P['호우등급'].eq(0) & P['일강수'].fillna(0).lt(RAIN_MAX)
P['fold'] = P['날짜'].dt.isocalendar().week.astype(int) % N_FOLDS
P = P.sort_values(['gkey', 'slot', '날짜']).reset_index(drop=True)
dd = P.drop_duplicates(['대상', '날짜'])
print('평시(학습) 날 수 — 월별'); display(dd[dd['평시']].groupby(['대상', dd['날짜'].dt.month]).size().unstack(fill_value=0))

# ======================================================================
# 5. 휴무 칸
# ======================================================================
zr = P[P['평시']].groupby(['skey', 'daytype'])['S'].apply(lambda s: (s == 0).mean()).rename('평시0비율')
P = P.join(zr, on=['skey', 'daytype'])
P['휴무'] = P['평시0비율'].fillna(0).ge(CLOSED_SHARE)
P = P.drop(columns='평시0비율')
print(f"휴무 칸 {P['휴무'].sum():,} ({P['휴무'].mean():.2%})")

# ======================================================================
# 6. 달력 변수 8개
# ======================================================================
cal = pd.DataFrame({'날짜': pd.date_range(P['날짜'].min() - pd.Timedelta(days=5), P['날짜'].max() + pd.Timedelta(days=5))})
off = (cal['날짜'].dt.dayofweek.ge(5) | cal['날짜'].isin(HOLI)).to_numpy()
blk, pos = np.zeros(len(cal), int), np.zeros(len(cal), int)
i = 0
while i < len(cal):
    if not off[i]: i += 1; continue
    j = i
    while j + 1 < len(cal) and off[j + 1]: j += 1
    blk[i:j + 1] = j - i + 1
    if j - i + 1 >= 3: pos[i] = 1; pos[j] = 3; pos[i + 1:j] = 2
    i = j + 1
cal['연휴위치'] = pos
cal['연휴전날'] = (~off & (np.r_[blk[1:], 0] >= 3)).astype(int)
cal['연휴다음날'] = (~off & (np.r_[0, blk[:-1]] >= 3)).astype(int)
d = cal['날짜'].dt
cal['월초'] = d.day.le(3).astype(int)
cal['월말'] = (d.days_in_month - d.day).le(6).astype(int)
cal['연말'] = cal['날짜'].between(*YEAR_END).astype(int)
cal['성탄이브'] = cal['날짜'].eq(XMAS_EVE).astype(int)
P = P.merge(cal, on='날짜', how='left').sort_values(['gkey', 'slot', '날짜']).reset_index(drop=True)

chk = P.drop_duplicates('날짜').set_index('날짜')
show_ = chk[(chk[['연휴위치', '연휴전날', '연휴다음날', '성탄이브']].sum(axis=1) > 0) | chk['추석직전'].eq(1)]
print('[확인] 연휴·명절·연말 표시가 있는 날'); display(show_[['공휴일', '추석직전', '연휴위치', '연휴전날', '연휴다음날', '연말', '성탄이브']].assign(요일=show_.index.day_name().str[:3]))

# ======================================================================
# 7. 주기 달력 4개 (유치원 2곳에만)
# ======================================================================
cal2 = pd.DataFrame({'날짜': pd.date_range(P['날짜'].min().replace(day=1), P['날짜'].max() + pd.offsets.MonthEnd(0))})
d = cal2['날짜'].dt
cal2['영업일'] = ~(d.dayofweek.ge(5) | cal2['날짜'].isin(HOLI))
ym = d.to_period('M')
cal2['일자'] = d.day
cal2['영업일순번'] = cal2.groupby(ym)['영업일'].cumsum().astype(int)
cal2['남은영업일'] = (cal2.groupby(ym)['영업일'].transform('sum') - cal2['영업일순번']).astype(int)
cal2['월내주차'] = ((d.day - 1) // 7 + 1).astype(int)
CYC_CAL = ['일자', '영업일순번', '남은영업일', '월내주차']
P = P.merge(cal2[['날짜'] + CYC_CAL], on='날짜', how='left').sort_values(['gkey', 'slot', '날짜']).reset_index(drop=True)

GK = P['gkey'].unique()
CYC_G = sorted(g for g in GK if any(g.split('|')[0].startswith(r) and p in g.split('|')[1] for r, p in CYCLE_INDS))
print('주기 달력 업종:', CYC_G)
if len(CYC_G) != len(CYCLE_INDS): print('⚠ 지정한 업종 중 데이터에 없는 업종이 있습니다(희소 칸으로 빠졌을 수 있음)')
ex = cal2[cal2['날짜'].between('2025-10-01', '2025-10-31')].assign(요일=lambda x: x['날짜'].dt.day_name().str[:3])
print('[예시] 2025년 10월'); display(ex.set_index('날짜')[['요일', '영업일'] + CYC_CAL].T)

# ======================================================================
# 8. V4: 공휴일을 일요일처럼 넣을 업종
# ======================================================================
SUN_G = sorted(g for g in GK if any(g.startswith(r) for r in HOLI_SUN_ALL_REGIONS)
               or any(g.split('|')[0].startswith(r) and g.split('|')[1] == u for r, u in HOLI_SUN_LIST))
miss = [u for r, u in HOLI_SUN_LIST if not any(g.split('|')[0].startswith(r) and g.split('|')[1] == u for g in GK)]
print(f'일요일 처리 업종 {len(SUN_G)}개 (강남 {sum(g.startswith("강남") for g in SUN_G)}, 춘천 {sum(g.startswith("춘천") for g in SUN_G)})')
print('춘천:', [g.split('|')[1] for g in SUN_G if g.startswith('춘천')])
if miss: print('⚠ 데이터에 없는 춘천 업종:', miss)

# ======================================================================
# 9. 변수 만들기 + 교차 적합 학습
# ======================================================================
def fold_masks(f, k):
    dates = f['날짜'].values.astype('datetime64[D]'); te = (f['fold'] == k).values
    near = np.zeros(len(f), bool)
    for off_ in range(-BUFFER, BUFFER + 1):
        near |= np.isin(dates, np.unique(dates[te]) + np.timedelta64(off_, 'D'))
    return ~near, te          # (학습 가능 행 = 예측 주 ±3일 밖, 예측 행)
FOLD_P = {k: fold_masks(P, k) for k in range(N_FOLDS)}
PLAIN = P['평시'].to_numpy()

def mad_ratio(x):
    m = x.median(); return (x - m).abs().median() / m if m > 0 else np.nan
def vol_feats(trn):
    g = P.loc[trn, ['skey', 'S']].groupby('skey')['S']
    T = pd.DataFrame({'과거중앙값': np.log1p(g.median()), 'MAD비율': g.apply(mad_ratio),
                      '매출0비율': g.apply(lambda x: (x == 0).mean()), '관측날수': g.size().astype(float)})
    out = T.reindex(P['skey']); out.index = P.index; out['관측날수'] = out['관측날수'].fillna(0)
    return out

CAL = ['요일', '주말', '공휴일', '월초', '월말', '추석직전', '연휴위치', '연휴전날', '연휴다음날', '연말', '성탄이브']
X0 = pd.DataFrame({'대상': P['대상'].astype('category'), '업종': P['업종'].astype('category'), '시간대': P['slot'].astype('category')}, index=P.index)
for c in CAL: X0[c] = P[c].astype(int)
# V4 공휴일 → 일요일
sun = P['공휴일'].eq(1).to_numpy() & P['gkey'].isin(SUN_G).to_numpy()
X0.loc[sun, '요일'] = 6; X0.loc[sun, '주말'] = 1
print(f'일요일로 바꾼 행 {sun.sum():,}개')
# 주기 달력: 유치원만 값, 나머지 NaN
cm = P['gkey'].isin(CYC_G).to_numpy()
XC = pd.DataFrame({f'주기_{c}': np.where(cm, P[c].astype(float), np.nan) for c in CYC_CAL}, index=P.index)

SW = P['S'].to_numpy(float) / SCALE
B = np.full(len(P), np.nan); MODELS = {}
t0 = time.time()
for k, (free, te) in FOLD_P.items():
    trn = free & PLAIN
    X = pd.concat([X0, vol_feats(trn), XC], axis=1)
    m = xgb.XGBRegressor(**XGB_PARAMS).fit(X.loc[trn], SW[trn])
    B[te] = m.predict(X.loc[te]) * SCALE
    MODELS[k] = m
    print(f'fold {k}: 학습 {trn.sum():,}행 / 예측 {te.sum():,}행 ({time.time() - t0:.0f}초)')
P['B'] = np.clip(B, 0, None)
FEATS = list(X.columns)
assert len(FEATS) == 22 and P['B'].notna().all()
print('변수', len(FEATS), '개:', FEATS)

# ======================================================================
# 10. 결과 저장
# ======================================================================
P['B_하루합'] = P.groupby(['gkey', '날짜'])['B'].transform('sum')
P['y'] = (P['S'] - P['B']) / P['B_하루합'].replace(0, np.nan)
P['특수일'] = P['날짜'].isin(SPECIAL_DAYS)
COLS = ['대상', '업종', 'gkey', 'slot', '날짜', 'S', 'N', 'B', 'B_하루합', 'y', '휴무', '평시', '특수일', '폭염등급', '호우등급',
        '일최고기온', '일강수', '시간대최고기온', '시간대강수', '시간대강수시간', '요일', '공휴일', 'fold']
OUT = P[COLS]
OUT_PATH = f'{OUT_DIR}/38_최종baseline_V4.csv'
OUT.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print('저장:', OUT_PATH, OUT.shape)
display(OUT.head())

# ======================================================================
# 11. 검증 요약 (평시 날, 휴무 칸 제외)
# ======================================================================
iqr = lambda s: s.quantile(.75) - s.quantile(.25)
E = P[P['평시'] & ~P['휴무']].copy()

def metrics(q):
    st = q.groupby('skey')['y'].agg(['mean', 'std', 'size']); t = st['mean'] / (st['std'] / np.sqrt(st['size']))
    den = q.drop_duplicates(['gkey', '날짜'])['B_하루합'].sum()
    S, Bv = q['S'].to_numpy(), q['B'].to_numpy(); ok = S > 0
    sh_s = q['S'] / q.groupby(['gkey', '날짜'])['S'].transform('sum').replace(0, np.nan)
    sh_b = q['B'] / q['B_하루합'].replace(0, np.nan)
    return {'① 금액가중 y(%)': (q['S'] - q['B']).sum() / den * 100, '① y 중앙값(%)': q['y'].median() * 100,
            '① |t|>2 칸 비율(%)': (t.abs() > 2).mean() * 100, '② 칸별 y IQR 중앙값(%p)': q.groupby('skey')['y'].agg(iqr).median() * 100,
            '④ 시간대 비중 오차(%p)': (sh_s - sh_b).abs().mean() * 100,
            '⑤ MdAPE(%)': np.median(np.abs(S[ok] - Bv[ok]) / S[ok]) * 100, '⑤ WAPE(%)': np.abs(S - Bv).sum() / S.sum() * 100}

rows = []
for reg in sorted(E['대상'].unique()):
    for lab, q in [('특수일 포함', E[E['대상'] == reg]), ('특수일 제외', E[(E['대상'] == reg) & ~E['특수일']])]:
        rows.append({'지역': reg, '범위': lab, **metrics(q)})
SUMM = pd.DataFrame(rows).set_index(['지역', '범위'])
print('[①②④⑤] 평시 지표'); display(SUMM.round(2))

print('[① 월별] 금액가중 y(%) — 특수일 제외')
q = E[~E['특수일']]
mon = q.groupby(['대상', q['날짜'].dt.month]).apply(lambda g: (g['S'] - g['B']).sum() / g.drop_duplicates(['gkey', '날짜'])['B_하루합'].sum() * 100)
display(mon.unstack().round(2))

# ③ 실제 사건 규모 합계: 지역별, 사건 날의 요일 구성과 같게 평시 날을 무작위로 뽑음
rng = np.random.default_rng(0)
day = E[~E['특수일']].groupby(['대상', '날짜']).agg(S=('S', 'sum'), B=('B', 'sum')).reset_index()
day['요일'] = day['날짜'].dt.dayofweek
ev = P.drop_duplicates(['대상', '날짜']).assign(요일=lambda x: x['날짜'].dt.dayofweek)
rows = []
for reg, dq in day.groupby('대상'):
    pools = {w: g for w, g in dq.groupby('요일')}
    for lab, col in [('폭염 시즌', '폭염등급'), ('호우 시즌', '호우등급')]:
        wd = ev[(ev['대상'] == reg) & (ev[col] > 0)]['요일'].tolist()
        if not wd: continue
        errs = []
        for _ in range(300):
            pick = pd.concat([pools[w].sample(1, random_state=int(rng.integers(1e9))) for w in wd if w in pools])
            errs.append((pick['S'].sum() / pick['B'].sum() - 1) * 100)
        errs = np.array(errs)
        rows.append({'지역': reg, '구간': f'{lab} ({len(wd)}일)', '합계 편향(%)': errs.mean(), '|오차| 평균(%)': np.abs(errs).mean(),
                     '90% 범위(%)': f'{np.percentile(errs, 5):+.1f} ~ {np.percentile(errs, 95):+.1f}'})
SEASON = pd.DataFrame(rows)
print('[③] 사건 규모 평시 합계 오차 (300번 무작위)'); display(SEASON.round(2))

print('[참고] 공휴일 하루 합계 편향(%) — 지역 전체, 휴무 칸 제외')
h = P[P['공휴일'].eq(1) & ~P['휴무']]
display(h.groupby(['대상', h['날짜'].dt.strftime('%m/%d')]).apply(lambda g: (g['S'].sum() / g['B'].sum() - 1) * 100).unstack().round(1))

SUMM.reset_index().to_csv(f'{OUT_DIR}/38_최종baseline_V4_검증요약.csv', index=False, encoding='utf-8-sig')
print('저장:', f'{OUT_DIR}/38_최종baseline_V4_검증요약.csv')

# ======================================================================
# 12. (선택) 36번 결과와 같은지 확인
# ======================================================================
hits = sorted(glob.glob(f'{OUT_DIR}/**/36_공휴일일요일처리_교차적합*.csv', recursive=True))
if not hits:
    print('36번 결과 파일이 없어 건너뜁니다.')
else:
    old = pd.read_csv(hits[0], encoding='utf-8-sig', parse_dates=['날짜'])
    col = next((c for c in old.columns if c.startswith('B_V4')), None)
    if col is None: print('36번 파일에 B_V4 열이 없습니다 (V4 추가 전 버전).')
    else:
        cmp_ = P[['gkey', 'slot', '날짜', 'B']].merge(old[['gkey', 'slot', '날짜', col]], on=['gkey', 'slot', '날짜'])
        rel = (cmp_['B'] - cmp_[col]).abs() / cmp_[col].abs().clip(lower=1)
        print(f'겹치는 행 {len(cmp_):,} / 상관 {np.corrcoef(cmp_["B"], cmp_[col])[0, 1]:.6f} / '
              f'상대차이 중앙값 {rel.median():.2e}, 최대 {rel.max():.2e} / 합계 비 {cmp_["B"].sum() / cmp_[col].sum():.6f}')
