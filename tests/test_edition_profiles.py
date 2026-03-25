"""Tests for edition profiles."""

from extracosmic_commons.edition_profiles import (
    get_profile,
    get_profiles_for_work,
    list_profiles,
    register_profile,
    EditionProfile,
)


class TestEditionProfiles:
    def test_get_builtin_profile(self):
        p = get_profile("di-giovanni-2010")
        assert p is not None
        assert p.work_id == "hegel-sol"
        assert p.reference_system == "gw"

    def test_get_nonexistent_profile(self):
        assert get_profile("nonexistent") is None

    def test_list_profiles(self):
        profiles = list_profiles()
        assert len(profiles) >= 9  # built-in count
        ids = [p.edition_id for p in profiles]
        assert "di-giovanni-2010" in ids
        assert "guyer-wood-1998" in ids

    def test_profiles_for_work(self):
        sol_profiles = get_profiles_for_work("hegel-sol")
        assert len(sol_profiles) >= 4  # di Giovanni, Miller, GW21, GW11
        ids = [p.edition_id for p in sol_profiles]
        assert "di-giovanni-2010" in ids
        assert "miller-1969" in ids

    def test_register_custom_profile(self):
        custom = EditionProfile(
            name="Test Profile",
            work_id="test-work",
            edition_id="test-edition",
            reference_system="custom",
            ref_label="T",
        )
        register_profile(custom)
        assert get_profile("test-edition") is not None
        assert get_profile("test-edition").work_id == "test-work"

    def test_to_source_metadata(self):
        p = get_profile("di-giovanni-2010")
        meta = p.to_source_metadata()
        assert meta["work_id"] == "hegel-sol"
        assert meta["edition_id"] == "di-giovanni-2010"
        assert meta["original_date"] == "1812"
        assert meta["edition_date"] == "2010"
        assert meta["translator"] == "di Giovanni, George"

    def test_hegel_margin_pattern(self):
        p = get_profile("di-giovanni-2010")
        assert p.margin_ref_pattern is not None
        m = p.margin_ref_pattern.search("See page 21.364 for details")
        assert m is not None
        assert m.group(1) == "21.364"

    def test_kant_margin_pattern(self):
        p = get_profile("guyer-wood-1998")
        m = p.margin_ref_pattern.search("This is at A 235")
        assert m is not None
        assert "A" in m.group(1)

    def test_original_and_edition_dates(self):
        p = get_profile("inwood-2018")
        assert p.original_date == "1807"
        assert p.edition_date == "2018"

    def test_version_info(self):
        p = get_profile("gw-21")
        assert p.version_id == "sol-being-1832"
        assert p.gw_volume == "21"
        assert p.is_original_language is True
