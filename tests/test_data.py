import pytest

from churn import data
from churn.data import clean


def test_blank_total_charges_become_zero_for_new_customers(raw):
    cleaned = clean(raw)
    new = cleaned[cleaned["tenure"] == 0]
    assert len(new) > 0
    assert (new["TotalCharges"] == 0).all()


def test_a_blank_total_for_an_existing_customer_is_refused(raw):
    raw = raw.copy()
    row = raw.index[raw["tenure"] > 0][0]
    raw.loc[row, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="tenure > 0"):
        clean(raw)


def test_target_and_senior_flag_are_encoded(raw):
    cleaned = clean(raw)
    assert set(cleaned["Churn"].unique()) <= {0, 1}
    assert set(cleaned["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_a_file_with_the_wrong_fingerprint_is_refused(tmp_path):
    (tmp_path / "Telco-Customer-Churn.csv").write_text("customerID,Churn\n1,No\n")
    with pytest.raises(ValueError, match="SHA-256"):
        data.fetch(tmp_path)


@pytest.mark.network
def test_the_pinned_dataset_loads_whole():
    df = data.load()
    assert len(df) == data.N_ROWS
    assert round(df["Churn"].mean(), 4) == 0.2654
    assert df["TotalCharges"].notna().all()
