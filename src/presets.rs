//! Named channels from `python/ionochannel/presets.json` (shared with the
//! Python package): ITU-R F.520 and F.1487.

use crate::channel::PathSpec;
use crate::fading::Component;
use serde_json::Value;

const JSON: &str = include_str!("../python/ionochannel/presets.json");

#[derive(Clone, Debug)]
pub struct Preset {
    pub name: String,
    pub source: String,
    pub paths: Vec<PathSpec>,
}

fn num(v: &Value, k: &str, default: Option<f64>) -> f64 {
    v.get(k)
        .and_then(Value::as_f64)
        .or(default)
        .unwrap_or_else(|| panic!("presets.json: missing {k} in {v}"))
}

fn component(v: &Value) -> Component {
    let power = num(v, "power", None);
    let shift_hz = num(v, "shift_hz", Some(0.0));
    if v.get("specular").and_then(Value::as_bool).unwrap_or(false) {
        Component::Specular { power, shift_hz }
    } else {
        Component::Scatter {
            power,
            spread_hz: num(v, "spread_hz", None),
            shift_hz,
        }
    }
}

fn path(v: &Value) -> PathSpec {
    let delay_s = num(v, "delay_ms", None) * 1e-3;
    let components = match v.get("components").and_then(Value::as_array) {
        Some(cs) => cs.iter().map(component).collect(),
        None => vec![component(v)],
    };
    PathSpec {
        delay_s,
        components,
    }
}

/// All presets, sorted by name.
pub fn all() -> Vec<Preset> {
    let v: Value = serde_json::from_str(JSON).expect("presets.json");
    v.as_object()
        .expect("presets.json: an object")
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, p)| Preset {
            name: k.clone(),
            source: p
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            paths: p["paths"]
                .as_array()
                .expect("paths")
                .iter()
                .map(path)
                .collect(),
        })
        .collect()
}

pub fn get(name: &str) -> Option<Preset> {
    all().into_iter().find(|p| p.name == name)
}
