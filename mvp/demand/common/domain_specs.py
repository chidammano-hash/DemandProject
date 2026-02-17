from dataclasses import dataclass


@dataclass(frozen=True)
class DomainSpec:
    name: str
    plural: str
    table: str
    ck_field: str
    business_key_field: str
    columns: list[str]
    source_file: str
    clean_file: str
    search_fields: list[str]
    int_fields: set[str]
    float_fields: set[str]
    date_fields: set[str]
    default_sort: str
    business_key_fields: tuple[str, ...] = ()
    business_key_separator: str = "-"
    source_delimiter: str = ","
    source_columns: dict[str, str] | None = None

    @property
    def columns_with_ck(self) -> list[str]:
        return [self.ck_field, *self.columns]

    @property
    def key_fields(self) -> tuple[str, ...]:
        return self.business_key_fields or (self.business_key_field,)

    def source_col_for(self, target_col: str) -> str:
        if not self.source_columns:
            return target_col
        return self.source_columns.get(target_col, target_col)


ITEM_SPEC = DomainSpec(
    name="item",
    plural="items",
    table="dim_item",
    ck_field="item_ck",
    business_key_field="item_no",
    columns=[
        "item_no",
        "item_desc",
        "item_status",
        "brand_name",
        "category",
        "class",
        "sub_class",
        "country",
        "scm_rtd_flag",
        "size",
        "case_weight",
        "cpl",
        "cpp",
        "lpp",
        "case_weight_uom",
        "bpc",
        "bottle_pack",
        "pack_case",
        "item_proof",
        "upc",
        "national_service_model",
        "supplier_no",
        "supplier_name",
        "item_is_deleted",
        "producer_name",
    ],
    source_file="itemdata.csv",
    clean_file="itemdata_clean.csv",
    search_fields=[
        "item_no",
        "item_desc",
        "brand_name",
        "category",
        "class",
        "sub_class",
        "country",
        "upc",
        "supplier_no",
        "supplier_name",
        "producer_name",
        "national_service_model",
    ],
    int_fields={"cpl", "cpp", "lpp", "bpc", "bottle_pack", "pack_case"},
    float_fields={"case_weight", "item_proof"},
    date_fields=set(),
    default_sort="item_no",
)

LOCATION_SPEC = DomainSpec(
    name="location",
    plural="locations",
    table="dim_location",
    ck_field="location_ck",
    business_key_field="location_id",
    columns=[
        "location_id",
        "site_id",
        "site_desc",
        "state_id",
        "primary_demand_location",
    ],
    source_file="locationdata.csv",
    clean_file="locationdata_clean.csv",
    search_fields=[
        "location_id",
        "site_id",
        "site_desc",
        "state_id",
        "primary_demand_location",
    ],
    int_fields=set(),
    float_fields=set(),
    date_fields=set(),
    default_sort="location_id",
)

CUSTOMER_SPEC = DomainSpec(
    name="customer",
    plural="customers",
    table="dim_customer",
    ck_field="customer_ck",
    business_key_field="customer_no",
    business_key_fields=("site", "customer_no"),
    columns=[
        "site",
        "customer_no",
        "customer_name",
        "city",
        "state",
        "zip",
        "premise_code",
        "status",
        "license_name",
        "store_type_desc",
        "chain_type_desc",
        "state_chain_name",
        "corp_chain_name",
        "rpt_channel_desc",
        "rpt_sub_channel_desc",
        "rpt_ship_type_desc",
        "customer_acct_type_desc",
        "delivery_freq_code",
    ],
    source_file="customerdata.csv",
    clean_file="customerdata_clean.csv",
    search_fields=[
        "site",
        "customer_no",
        "customer_name",
        "city",
        "state",
        "zip",
        "status",
        "store_type_desc",
        "chain_type_desc",
        "state_chain_name",
        "corp_chain_name",
        "rpt_channel_desc",
        "rpt_sub_channel_desc",
        "rpt_ship_type_desc",
        "customer_acct_type_desc",
        "delivery_freq_code",
    ],
    int_fields=set(),
    float_fields=set(),
    date_fields=set(),
    default_sort="customer_ck",
)

TIME_SPEC = DomainSpec(
    name="time",
    plural="times",
    table="dim_time",
    ck_field="time_ck",
    business_key_field="date_key",
    columns=[
        "date_key",
        "day_name",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "iso_week_year",
        "iso_week",
        "week_start_date",
        "week_end_date",
        "month_number",
        "month_name",
        "month_start_date",
        "month_end_date",
        "quarter_number",
        "quarter_label",
        "quarter_start_date",
        "quarter_end_date",
        "year_number",
        "year_start_date",
        "year_end_date",
        "week_bucket",
        "month_bucket",
        "quarter_bucket",
        "year_bucket",
    ],
    source_file="_generated_time_2020_2035",
    clean_file="timedata_clean.csv",
    search_fields=[
        "date_key",
        "day_name",
        "month_name",
        "week_bucket",
        "month_bucket",
        "quarter_bucket",
        "year_bucket",
    ],
    int_fields={
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "iso_week_year",
        "iso_week",
        "month_number",
        "quarter_number",
        "year_number",
    },
    float_fields=set(),
    date_fields={
        "date_key",
        "week_start_date",
        "week_end_date",
        "month_start_date",
        "month_end_date",
        "quarter_start_date",
        "quarter_end_date",
        "year_start_date",
        "year_end_date",
    },
    default_sort="date_key",
)

DFU_SPEC = DomainSpec(
    name="dfu",
    plural="dfus",
    table="dim_dfu",
    ck_field="dfu_ck",
    business_key_field="dmdunit",
    business_key_fields=("dmdunit", "dmdgroup", "loc"),
    business_key_separator="_",
    columns=[
        "dmdunit",
        "dmdgroup",
        "loc",
        "brand",
        "abc_vol",
        "brand_desc",
        "ded_div_sw",
        "execution_lag",
        "otc_status",
        "premise",
        "prod_subgrp_desc",
        "region",
        "service_lvl_grp",
        "size",
        "state_plan",
        "supergroup",
        "supplier_desc",
        "total_lt",
        "vintage",
        "sales_div",
        "purge_sw",
        "alcoh_pct",
        "bot_type_desc",
        "brand_size",
        "cnty",
        "dom_imp_opt",
        "grape_vrty_desc",
        "material",
        "prod_cat_desc",
        "producer_desc",
        "proof",
        "subclass_desc",
        "prod_class_desc",
        "file_dt",
        "histstart",
        "cluster_assignment",
        "ml_cluster",
        "sop_ref",
    ],
    source_file="dfu.txt",
    clean_file="dfu_clean.csv",
    search_fields=[
        "dmdunit",
        "loc",
        "brand",
        "brand_desc",
        "region",
        "state_plan",
        "sales_div",
        "prod_cat_desc",
        "prod_class_desc",
        "subclass_desc",
        "supplier_desc",
        "producer_desc",
        "cluster_assignment",
        "ml_cluster",
        "dmdgroup",
    ],
    int_fields={"ded_div_sw", "execution_lag", "total_lt", "vintage", "purge_sw"},
    float_fields={"alcoh_pct", "proof"},
    date_fields=set(),
    default_sort="dmdunit",
    source_delimiter="|",
    source_columns={
        "abc_vol": "U_ABC_VOL",
        "brand_desc": "U_BRAND_DESC",
        "ded_div_sw": "U_DED_DIV_SW",
        "execution_lag": "U_EXECUTION_LAG",
        "otc_status": "U_OTC_STATUS",
        "premise": "U_PREMISE",
        "prod_subgrp_desc": "U_PROD_SUBGRP_DESC",
        "region": "U_REGION",
        "service_lvl_grp": "U_SERVICE_LVL_GRP",
        "size": "U_SIZE",
        "state_plan": "U_STATE_PLAN",
        "supergroup": "U_SUPERGROUP",
        "supplier_desc": "U_SUPPLIER_DESC",
        "total_lt": "U_TOTAL_LT",
        "vintage": "U_VINTAGE",
        "sales_div": "U_SALES_DIV",
        "purge_sw": "U_PURGE_SW",
        "alcoh_pct": "U_ALCOH_PCT",
        "bot_type_desc": "U_BOT_TYPE_DESC",
        "brand_size": "U_BRAND_SIZE",
        "cnty": "U_CNTY",
        "dom_imp_opt": "U_DOM_IMP_OPT",
        "grape_vrty_desc": "U_GRAPE_VRTY_DESC",
        "material": "U_MATERIAL",
        "prod_cat_desc": "U_PROD_CAT_DESC",
        "producer_desc": "U_PRODUCER_DESC",
        "proof": "U_PROOF",
        "subclass_desc": "U_SUBCLASS_DESC",
        "prod_class_desc": "U_PROD_CLASS_DESC",
        "cluster_assignment": "U_CLUSTER_ASSIGNMENT",
        "sop_ref": "U_SOP_REF",
    },
)

SALES_SPEC = DomainSpec(
    name="sales",
    plural="sales",
    table="fact_sales_monthly",
    ck_field="sales_ck",
    business_key_field="dmdunit",
    business_key_fields=("dmdunit", "dmdgroup", "loc", "startdate", "type"),
    business_key_separator="_",
    columns=[
        "dmdunit",
        "dmdgroup",
        "loc",
        "startdate",
        "type",
        "qty_shipped",
        "qty_ordered",
        "qty",
        "file_dt",
    ],
    source_file="dfu_lvl2_hist.txt",
    clean_file="dfu_lvl2_hist_clean.csv",
    search_fields=["dmdunit", "dmdgroup", "loc", "startdate", "type", "file_dt"],
    int_fields={"type"},
    float_fields={"qty_shipped", "qty_ordered", "qty"},
    date_fields={"startdate", "file_dt"},
    default_sort="startdate",
    source_delimiter="|",
    source_columns={
        "qty_shipped": "U_QTY_SHIPPED",
        "qty_ordered": "U_QTY_ORDERED",
    },
)

FORECAST_SPEC = DomainSpec(
    name="forecast",
    plural="forecasts",
    table="fact_external_forecast_monthly",
    ck_field="forecast_ck",
    business_key_field="dmdunit",
    business_key_fields=("dmdunit", "dmdgroup", "loc", "fcstdate", "startdate"),
    business_key_separator="_",
    columns=[
        "dmdunit",
        "dmdgroup",
        "loc",
        "fcstdate",
        "startdate",
        "lag",
        "execution_lag",
        "basefcst_pref",
        "tothist_dmd",
        "model_id",
    ],
    source_file="dfu_stat_fcst.txt",
    clean_file="dfu_stat_fcst_clean.csv",
    search_fields=["dmdunit", "dmdgroup", "loc", "fcstdate", "startdate", "lag", "execution_lag", "model_id"],
    int_fields={"lag", "execution_lag"},
    float_fields={"basefcst_pref", "tothist_dmd"},
    date_fields={"fcstdate", "startdate"},
    default_sort="fcstdate",
    source_delimiter="|",
)


DOMAIN_SPECS: dict[str, DomainSpec] = {
    ITEM_SPEC.name: ITEM_SPEC,
    LOCATION_SPEC.name: LOCATION_SPEC,
    CUSTOMER_SPEC.name: CUSTOMER_SPEC,
    TIME_SPEC.name: TIME_SPEC,
    DFU_SPEC.name: DFU_SPEC,
    SALES_SPEC.name: SALES_SPEC,
    FORECAST_SPEC.name: FORECAST_SPEC,
}


def get_spec(name: str) -> DomainSpec:
    key = (name or "").strip().lower()
    if key not in DOMAIN_SPECS:
        allowed = ", ".join(sorted(DOMAIN_SPECS))
        raise ValueError(f"Unknown dataset '{name}'. Allowed: {allowed}")
    return DOMAIN_SPECS[key]
