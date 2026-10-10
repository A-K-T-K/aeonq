"""The saved benchmark evidence must still match the pinned manifest."""

from benchmarks.pipeline.audit_reproducibility import audit


def test_reproducibility_audit_accepts_utf8_bom_lock_and_all_trials():
    report = audit()
    assert report["passed"] is True
    assert report["trial_counts"] == {
        "publication_reference": 23_800,
        "publication_cross_sdk": 800,
        "publication_scheduler_qiskit": 750,
        "publication_case_study": 2_400,
        "publication_case_study_sdk": 900,
        "publication_case_study_qsharp": 45,
    }
    assert report["installed_dependency_mismatches"] == []
