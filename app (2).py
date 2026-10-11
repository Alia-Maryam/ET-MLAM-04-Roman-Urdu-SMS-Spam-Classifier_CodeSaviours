"""
Project 4: Logistic Regression - Roman Urdu SMS Spam Classifier  (Streamlit version)

Hand-crafted features (length, digits, caps ratio, spam words, links, phone numbers, "!")
-> StandardScaler -> logistic regression. Pure NumPy/pandas: no scikit-learn needed.

Run locally:   streamlit run app.py
Deploy:        push app.py + requirements.txt (+ roman_urdu_spam_dataset.csv) to GitHub,
               then deploy on Streamlit Cloud.
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA_FILE = "roman_urdu_spam_dataset.csv"

# Words that recur in the spam messages (same list as the notebook; editable in the sidebar)
DEFAULT_SPAM_WORDS = [
    "free", "jeeta", "jeet", "prize", "claim", "offer", "sale", "discount",
    "win", "winner", "lucky", "urgent", "verify", "block", "suspend",
    "click", "link", "loan", "cash", "guaranteed", "limited", "congratulations",
    "mubarak", "bank", "account", "pin", "otp", "customs",
]

FEATURE_NAMES = [
    "length", "digit_count", "caps_ratio", "spam_word_count",
    "has_link", "has_phone_number", "exclamation_count",
]

# The notebook's example messages
EXAMPLES = [
    "Kal party hai mere ghar, 7 baje aa jana",
    "Aap ne 200000 jeeta hai, is number par call karein 03009876543",
    "Bhai assignment submit karna mat bhoolna kal",
    "URGENT: aapka account 24 ghante mein band ho jayega, verify karein abhi",
    "Mummy ne bola hai raat ko jaldi ana ghar",
]

# ---------------------------------------------------------------------------
# Small demo dataset, used only when no CSV is available (NOT your real dataset)
# ---------------------------------------------------------------------------
DEMO_HAM = """Bhai kal cricket khelne chalna hai, ground par 5 baje
Ammi ki tabiyat ab theek hai, dawai se araam a gaya
Mujhe airport se pick karne aa jana, flight 6 baje land karegi
Naye semester ka schedule aa gaya hai, check kar lena
Kal exam hai, shaam 5 baje par pohanch jana
Assalam o alaikum, aaj ka lecture kitne baje hai?
Dinner mein biryani bana rahi hoon, jaldi ghar aao
Abbu ne kaha hai ke weekend par Lahore chalna hai
Meri kitaab wapas kar dena jab free ho
Raat ko chai pe milte hain, thori der ke liye
Office se nikal gaya hoon, 20 minute mein pohanch jaunga
Tumhari presentation kal subah 10 baje hai, tayyari kar lo
Dadi jaan ko dawai dene ka waqt ho gaya hai
Bazaar se doodh aur roti le aana please
Match dekhne mere ghar aa jao, sab dost aa rahe hain
Class cancel ho gayi hai aaj, ghar wapas ja sakte ho
Shukriya bhai, tumne bohat madad ki
Kal subah 8 baje university ki bus miss mat karna
Behan ki shaadi ki tareekh pakki ho gayi hai, 15 march
Mujhe notes bhej do, kal test hai
Chhutti mein Murree ka plan bana rahe hain, tum bhi chalo
Doctor ne kaha hai ke aaram karo aur paani zyada piyo
Meeting 3 baje shuru hogi, room number 12 mein
Gaari mein petrol khatam hone wala hai, pump par ruk jana
Eid par sab ghar aa rahe hain, tayyari shuru kar do
Assignment ki deadline Friday tak barh gayi hai
Aaj mausam bohat acha hai, park chalein?
Bijli ka bill jama kara diya hai, rasid mil gayi
Uncle ka phone aaya tha, unhone kal bulaya hai
Parhai ke baad library mein milte hain
Mubarak ho bhai, tumhari job lag gayi!
Bhai raat ko late ho jaunga, intezar mat karna
Teacher ne kaha hai project group mein karna hai
Subah ki namaz ke baad walk par chalte hain
Sabzi mandi se tamatar aur pyaz le aana
Cricket match ka score kya hua?
Naya phone kaisa chal raha hai?
Kal ki class ke notes WhatsApp kar dena
Abu ko bata dena main thori der mein aa raha hoon
Hum sab kal picnic par ja rahe hain, tum bhi aa jao
Exam ki tayyari kaisi chal rahi hai?
Mama ne khana bhej diya hai, le jana
Tumhara bag mere paas reh gaya tha, kal de dunga
Aaj office mein bohat kaam tha, thak gaya hoon"""

DEMO_SPAM = """SALE! 60% off har item par, sirf is weekend tak, order karein abhi
Mubarak ho! Aap ka number lucky draw mein select hua hai, inam claim karne ke liye call karein 03001234567
URGENT: Aap ka bank account block ho jayega, abhi link par click karein http://bank-verify.com
Free recharge hasil karein, abhi click karein www.freeload.com
Aap ne 5 lakh ka prize jeeta hai! Claim karne ke liye OTP bhejein
Guaranteed loan sirf 1 ghante mein, bina kisi guarantor ke, call 03451234567
Limited offer! 50% discount sirf aaj, jaldi karein!!!
Aapka PIN verify karein warna account suspend ho jayega
Congratulations! Aap winner hain, cash prize lene ke liye link par click karein
Customs ne aapka parcel rok liya hai, fees jama karwane ke liye 03111234567 par rabta karein
FREE FREE FREE! Mobile package par bumper discount, abhi activate karein
Aap ki account ki security khatre mein hai, urgent verify karein www.secure-bank.com
Lucky winner! Aap ne motorcycle jeeti hai, claim karein 03219876543
Cash loan asan qiston par, aaj hi apply karein, call 03331112233
Jeeta jeeta! Aap ka naam inami scheme mein aya hai, OTP share karein
BIG SALE! Sirf 3 din ke liye, 70% discount har kapre par!!!
Aap ka SIM block ho raha hai, verify karne ke liye abhi call karein 03001112222
Mubarak ho, aap ne iPhone jeeta hai, claim link: http://iphone-win.com
Limited time offer: free delivery aur discount code WIN50, click karein
Aap ki bank statement ready hai, link par click karke account verify karein
Congratulations! Aap ko 1000000 rupay ka inam mila hai, account number bhejein
URGENT! Aap ka account suspend hone wala hai, abhi PIN update karein
Guaranteed cash 50000 sirf CNIC par, abhi apply karein 03451239876
Free internet 100GB, offer limited hai, jaldi click karein www.netfree.com
Special offer! Har kharid par 40% discount, abhi visit karein shop.com
Aap ki shipment customs mein phansi hai, jaldi 3500 jama karwayein
Winner winner! Aap ka lucky number 7788 select hua, claim karein
Aap ke account se ghair mamooli activity hui hai, abhi link par verify karein
Sasta loan, kam markup, guaranteed approval, call 03008765432
Eid sale! Flat 50% discount, order abhi karein, limited stock!!!
Mubarak ho! Aap ko free ticket mil gaya hai, claim karne ke liye reply karein
BANK ALERT: aap ka account lock hai, unlock ke liye click karein www.bank-alert.com
Prize claim karne ki aakhri tareekh aaj hai, jaldi karein!!
Aap ne 25 lakh jeete hain, apna account PIN bhejein
Free gift voucher, offer sirf aaj ke liye, click karein
Loan chahiye? Cash fori hasil karein, call 03459998877
Verify your account abhi, warna block ho jayega, link: http://verify-now.com
Lucky customer! Discount coupon jeetein, abhi claim karein!!
Aap ka parcel customs par hai, 5000 rupay jama karein, call 03211239988
Congratulations winner! Free mobile jeetne ke liye click karein
Aap ki pension account verify karein warna suspend ho jayega
SUPER OFFER!!! Sirf aaj free delivery aur 60% discount
Account block hone se bachne ke liye turant PIN bhejein 1234
Cash prize 100000 jeetne ka mauka, abhi claim karein www.cashwin.com
Mubarak! Aap ko loan guaranteed manzoor hua hai, call 03004445566"""


def demo_dataframe():
    rows = [(m, 0) for m in DEMO_HAM.splitlines()] + [(m, 1) for m in DEMO_SPAM.splitlines()]
    return pd.DataFrame(rows, columns=["message", "label"])


# ---------------------------------------------------------------------------
# Feature engineering (identical logic to the notebook)
# ---------------------------------------------------------------------------
def extract_features(message, spam_words):
    msg = str(message)
    lower = msg.lower()

    length = len(msg)
    digit_count = sum(c.isdigit() for c in msg)
    letters = [c for c in msg if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in msg) / len(letters)) if letters else 0.0
    spam_word_count = sum(word in lower for word in spam_words)
    has_link = int(("http" in lower) or ("www" in lower) or (".com" in lower))

    # crude phone-number pattern: a run of 7+ consecutive digits
    digit_run = max_digit_run = 0
    for c in msg:
        if c.isdigit():
            digit_run += 1
            max_digit_run = max(max_digit_run, digit_run)
        else:
            digit_run = 0
    has_phone_number = int(max_digit_run >= 7)
    exclamation_count = msg.count("!")

    return [length, digit_count, caps_ratio, spam_word_count, has_link, has_phone_number, exclamation_count]


def feature_matrix(messages, spam_words):
    return np.array([extract_features(m, spam_words) for m in messages], dtype=float)


# ---------------------------------------------------------------------------
# Modelling helpers (NumPy only)
# ---------------------------------------------------------------------------
def approximate_mode(class_counts, n_draws, rng):
    """How many items to draw from each class (same rule scikit-learn uses for stratified splits)."""
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous).astype(int)
    need = int(n_draws - floored.sum())
    if need > 0:
        remainder = continuous - floored
        for value in np.sort(np.unique(remainder))[::-1]:
            inds = np.where(remainder == value)[0]
            add = min(len(inds), need)
            floored[rng.choice(inds, size=add, replace=False)] += 1
            need -= add
            if need == 0:
                break
    return floored


def stratified_split(y, test_size, seed):
    """Stratified train/test indices, matching train_test_split(..., stratify=y, random_state=seed)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_test = int(np.ceil(test_size * n))
    n_train = n - n_test
    classes, y_idx = np.unique(y, return_inverse=True)
    counts = np.bincount(y_idx)
    class_indices = np.split(np.argsort(y_idx, kind="mergesort"), np.cumsum(counts)[:-1])

    n_i = approximate_mode(counts, n_train, rng)
    t_i = approximate_mode(counts - n_i, n_test, rng)
    train, test = [], []
    for i in range(len(classes)):
        perm = rng.permutation(counts[i])
        picked = class_indices[i].take(perm, mode="clip")
        train.extend(picked[: n_i[i]])
        test.extend(picked[n_i[i]: n_i[i] + t_i[i]])
    return rng.permutation(train), rng.permutation(test)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(z / 2.0))  # numerically stable


def fit_logreg(X, y, C=1.0, max_iter=100, tol=1e-10):
    """L2-regularised logistic regression (Newton's method), same objective as scikit-learn's default."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    reg = np.ones(d + 1)
    reg[-1] = 0.0  # don't penalise the intercept
    w = np.zeros(d + 1)

    def loss(w_):
        z = Xb @ w_
        return C * np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(reg * w_ * w_)

    for _ in range(max_iter):
        p = sigmoid(Xb @ w)
        grad = C * Xb.T @ (p - y) + reg * w
        hess = C * (Xb.T * (p * (1 - p))) @ Xb + np.diag(reg) + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        f0, t = loss(w), 1.0
        while loss(w - t * step) > f0 and t > 1e-8:
            t /= 2
        w_new = w - t * step
        done = np.max(np.abs(w_new - w)) < tol
        w = w_new
        if done:
            break
    return w[:-1], w[-1]


def classification_report_df(y_true, y_pred):
    rows = []
    for cls, name in [(0, "Not Spam"), (1, "Spam")]:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"Class": name, "Precision": prec, "Recall": rec, "F1": f1, "Support": int((y_true == cls).sum())})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def train(df, spam_words, test_size, seed, C):
    X = feature_matrix(df["message"], spam_words)
    y = df["label"].to_numpy(dtype=int)

    train_idx, test_idx = stratified_split(y, test_size, seed)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0  # same as StandardScaler for constant columns

    coef, intercept = fit_logreg((X_train - mean) / scale, y_train.astype(float), C=C)
    model = {"mean": mean, "scale": scale, "coef": coef, "intercept": intercept, "spam_words": spam_words}

    y_pred = (predict_proba(model, X_test) >= 0.5).astype(int)
    cm = np.array([[int(((y_test == a) & (y_pred == b)).sum()) for b in (0, 1)] for a in (0, 1)])
    return {
        "X": X, "y": y, "model": model, "cm": cm,
        "accuracy": float((y_pred == y_test).mean()),
        "report": classification_report_df(y_test, y_pred),
        "n_train": len(train_idx), "n_test": len(test_idx),
    }


def predict_proba(model, X):
    """Probability of spam."""
    return sigmoid(((X - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Roman Urdu SMS Spam Classifier", page_icon="📩", layout="wide")
st.title("📩 Roman Urdu SMS Spam Classifier")
st.caption("Logistic regression on hand-crafted text features. Type a message and see why it is (or isn't) spam.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload dataset CSV", type="csv", help="Columns: message, label (1 = spam, 0 = not spam)")
    st.header("Model settings")
    test_size = st.slider("Test set share", 0.1, 0.4, 0.25, step=0.05)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    C = st.select_slider("Regularisation strength C (higher = weaker)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
    st.header("Spam words")
    words_text = st.text_area("One word per line (feeds the spam_word_count feature)", "\n".join(DEFAULT_SPAM_WORDS), height=220)

spam_words = tuple(w.strip().lower() for w in words_text.splitlines() if w.strip())

# ----- Load data -----
using_demo = False
try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        origin = f"uploaded file ({uploaded.name})"
    elif os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        origin = DATA_FILE
    else:
        df = demo_dataframe()
        origin = "built-in demo data"
        using_demo = True
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if not {"message", "label"}.issubset(df.columns):
    st.error("The CSV needs two columns named `message` and `label`.")
    st.stop()

df = df[["message", "label"]].dropna().reset_index(drop=True)
df["label"] = df["label"].astype(int)
if set(df["label"].unique()) != {0, 1}:
    st.error("`label` must contain both 0 (not spam) and 1 (spam).")
    st.stop()
if not spam_words:
    st.warning("The spam-word list is empty, so spam_word_count will always be 0.")

if using_demo:
    st.warning(
        f"No `{DATA_FILE}` found, so this is running on a small **demo dataset** (not your real one). "
        "Upload your CSV in the sidebar, or add it to the repo next to app.py."
    )

res = train(df, spam_words, test_size, int(seed), float(C))
model = res["model"]

tab_data, tab_feat, tab_model, tab_try = st.tabs(["📦 Data", "🧪 Features", "📊 Model", "✉️ Try a message"])

# ----- Data -----
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Messages", len(df))
    c2.metric("Spam", int((df["label"] == 1).sum()))
    c3.metric("Not spam", int((df["label"] == 0).sum()))
    st.caption(f"Source: {origin}")
    show = df.assign(label_name=np.where(df["label"] == 1, "Spam", "Not spam"))
    st.dataframe(show[["message", "label", "label_name"]], width="stretch", hide_index=True)

# ----- Features -----
with tab_feat:
    st.write("Each message becomes 7 numbers:")
    st.markdown(
        "- **length**: characters\n- **digit_count**: digits\n- **caps_ratio**: share of letters that are capitals\n"
        "- **spam_word_count**: how many spam words appear inside the text\n- **has_link**: `http`, `www` or `.com`\n"
        "- **has_phone_number**: a run of 7+ digits\n- **exclamation_count**: number of `!`"
    )
    feats = pd.DataFrame(res["X"], columns=FEATURE_NAMES)
    feats.insert(0, "label", np.where(res["y"] == 1, "Spam", "Not spam"))
    st.subheader("Average feature value: spam vs. not spam")
    st.dataframe(feats.groupby("label").mean().round(3), width="stretch")
    st.subheader("First rows of the feature matrix")
    st.dataframe(pd.concat([df[["message"]], feats], axis=1).head(15), width="stretch", hide_index=True)
    st.caption("Note: spam words match anywhere inside the text, so a short word like 'pin' also matches longer words that contain it.")

# ----- Model -----
with tab_model:
    st.write(f"Trained on **{res['n_train']}** messages, tested on **{res['n_test']}** (stratified, so both sets keep the spam/not-spam balance).")
    st.metric("Test accuracy", f"{res['accuracy']:.1%}")
    st.subheader("Classification report")
    rep = res["report"]
    st.dataframe(
        rep.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
        width="stretch", hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = res["cm"]
        cm_df = pd.DataFrame(
            [{"Actual": a, "Predicted": p, "Count": int(cm[i, j])}
             for i, a in enumerate(["Not Spam", "Spam"]) for j, p in enumerate(["Not Spam", "Spam"])]
        )
        base = alt.Chart(cm_df).encode(x=alt.X("Predicted:N", sort=["Not Spam", "Spam"]), y=alt.Y("Actual:N", sort=["Not Spam", "Spam"]))
        heat = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
        text = base.mark_text(fontSize=22).encode(text="Count:Q")
        st.altair_chart(heat + text, width="stretch")
    with right:
        st.subheader("Feature weights")
        coefs = pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": model["coef"]}).sort_values("coefficient", ascending=False)
        st.bar_chart(coefs.set_index("feature"))
        st.caption("Features are standardised first, so weights are comparable. Positive pushes toward spam, negative toward not spam.")
    st.dataframe(coefs.round(3), width="stretch", hide_index=True)
    st.info("If the test accuracy is 100%, remember the test set is small. That doesn't prove the model will be perfect on new messages.")

# ----- Try it -----
with tab_try:
    st.subheader("Check a message")
    default_msg = EXAMPLES[3]
    message = st.text_area("Message", value=default_msg, height=100)
    if message.strip():
        feats_new = feature_matrix([message], spam_words)
        p_spam = float(predict_proba(model, feats_new)[0])
        is_spam = p_spam >= 0.5
        a, b = st.columns(2)
        a.metric("Verdict", "🚨 SPAM" if is_spam else "✅ NOT SPAM")
        b.metric("Confidence", f"{(p_spam if is_spam else 1 - p_spam):.1%}")
        st.progress(min(max(p_spam, 0.0), 1.0), text=f"Spam probability: {p_spam:.1%}")

        scaled = (feats_new - model["mean"]) / model["scale"]
        contrib = pd.DataFrame(
            {"feature": FEATURE_NAMES, "value": feats_new[0].round(3), "pushes toward spam": (scaled[0] * model["coef"]).round(3)}
        )
        st.write("**Why?** Each feature's value and how strongly it pushed the verdict (positive = toward spam):")
        st.dataframe(contrib, width="stretch", hide_index=True)
        found = [w for w in spam_words if w in message.lower()]
        if found:
            st.caption("Spam words found: " + ", ".join(found))

    st.divider()
    st.subheader("Example messages from the notebook")
    ex_feats = feature_matrix(EXAMPLES, spam_words)
    ex_p = predict_proba(model, ex_feats)
    st.dataframe(
        pd.DataFrame(
            {"message": EXAMPLES,
             "verdict": np.where(ex_p >= 0.5, "SPAM", "NOT SPAM"),
             "confidence": [f"{(p if p >= 0.5 else 1 - p):.1%}" for p in ex_p]}
        ),
        width="stretch", hide_index=True,
    )
