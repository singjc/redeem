// Re-export Report and ReportSection from the external `report_builder` crate
// This keeps the internal module path the same while letting us depend on
// the shared `report-builder` package maintained separately.

pub use report_builder::Report;
pub use report_builder::ReportSection;

// Note: the original local implementation also provided convenience methods
// for adding plots and content blocks; the external crate exposes similar
// APIs. If there are API surface differences we can add small adapter
// functions here to preserve backward compatibility.
