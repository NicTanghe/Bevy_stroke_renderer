use alloc::collections::VecDeque;
use std::collections::{HashMap, HashSet};

use crate::{
    EffectDescriptor, EffectDomain, EffectId, EffectInfluence, EffectRegistry, PaintPlaneDescriptor,
};

/// Production no-op node. It exercises the same graph and tile path as later blur.
pub const PASS_THROUGH_EFFECT_ID: EffectId = EffectId(0);

/// Stable identity of one configured effect node in a document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectNodeId(pub u64);

/// Versioned document record for one non-destructive effect invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectNode {
    /// Identity used by dependency edges.
    pub id: EffectNodeId,
    /// Registered effect implementation.
    pub effect: EffectId,
    /// Implementation version persisted with the document.
    pub implementation_version: u32,
    /// Disabled nodes stay serializable but do not influence invalidation.
    pub enabled: bool,
    /// Versioned, implementation-owned non-destructive parameters.
    pub parameters: Vec<u8>,
}

/// Ordered dependency graph for material and display-surface effects.
#[derive(Clone, Debug)]
pub struct EffectGraph {
    revision: u64,
    next_node: u64,
    nodes: Vec<EffectNode>,
    dependencies: HashMap<EffectNodeId, Vec<EffectNodeId>>,
}

impl Default for EffectGraph {
    fn default() -> Self {
        Self {
            revision: 1,
            next_node: 1,
            nodes: vec![EffectNode {
                id: EffectNodeId(0),
                effect: PASS_THROUGH_EFFECT_ID,
                implementation_version: 1,
                enabled: true,
                parameters: Vec::new(),
            }],
            dependencies: HashMap::new(),
        }
    }
}

impl EffectGraph {
    /// Revision used to invalidate derived effect surfaces.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Configured nodes in stable document order.
    pub fn nodes(&self) -> &[EffectNode] {
        &self.nodes
    }

    pub(crate) fn persistence_parts(
        &self,
    ) -> (
        u64,
        u64,
        &[EffectNode],
        &HashMap<EffectNodeId, Vec<EffectNodeId>>,
    ) {
        (
            self.revision,
            self.next_node,
            &self.nodes,
            &self.dependencies,
        )
    }

    pub(crate) fn from_persistence_parts(
        revision: u64,
        next_node: u64,
        nodes: Vec<EffectNode>,
        dependencies: HashMap<EffectNodeId, Vec<EffectNodeId>>,
    ) -> Result<Self, &'static str> {
        let mut ids = HashSet::new();
        if nodes.iter().any(|node| !ids.insert(node.id)) {
            return Err("effect graph contains duplicate node ids");
        }
        if dependencies.iter().any(|(node, dependencies)| {
            !ids.contains(node)
                || dependencies
                    .iter()
                    .any(|dependency| !ids.contains(dependency))
        }) {
            return Err("effect graph contains an unknown dependency id");
        }
        let graph = Self {
            revision: revision.max(1),
            next_node: next_node.max(
                nodes
                    .iter()
                    .map(|node| node.id.0.saturating_add(1))
                    .max()
                    .unwrap_or(1),
            ),
            nodes,
            dependencies,
        };
        graph
            .topological_order()
            .is_some()
            .then_some(graph)
            .ok_or("effect graph contains a cycle")
    }

    /// Appends a registered node and returns its document identity.
    pub fn add_node(&mut self, effect: EffectId, implementation_version: u32) -> EffectNodeId {
        self.add_node_with_parameters(effect, implementation_version, Vec::new())
    }

    /// Appends a registered node with persisted implementation parameters.
    pub fn add_node_with_parameters(
        &mut self,
        effect: EffectId,
        implementation_version: u32,
        parameters: Vec<u8>,
    ) -> EffectNodeId {
        let id = EffectNodeId(self.next_node);
        self.next_node = self.next_node.wrapping_add(1);
        self.nodes.push(EffectNode {
            id,
            effect,
            implementation_version,
            enabled: true,
            parameters,
        });
        self.revision = self.revision.wrapping_add(1);
        id
    }

    /// Adds a `dependency -> node` edge, rejecting cycles and unknown nodes.
    pub fn add_dependency(
        &mut self,
        node: EffectNodeId,
        dependency: EffectNodeId,
    ) -> Result<(), &'static str> {
        if !self.contains(node) || !self.contains(dependency) {
            return Err("effect dependency references an unknown node");
        }
        if node == dependency {
            return Err("effect node cannot depend on itself");
        }
        let list = self.dependencies.entry(node).or_default();
        if list.contains(&dependency) {
            return Ok(());
        }
        list.push(dependency);
        if self.topological_order().is_none() {
            self.dependencies
                .get_mut(&node)
                .expect("dependency list was inserted above")
                .retain(|candidate| *candidate != dependency);
            return Err("effect dependency would create a cycle");
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Enables or disables a node without deleting its versioned parameters.
    pub fn set_enabled(&mut self, id: EffectNodeId, enabled: bool) -> bool {
        let Some(node) = self.nodes.iter_mut().find(|node| node.id == id) else {
            return false;
        };
        if node.enabled != enabled {
            node.enabled = enabled;
            self.revision = self.revision.wrapping_add(1);
        }
        true
    }

    /// Stable topological order, or `None` when the graph contains a cycle.
    pub fn topological_order(&self) -> Option<Vec<EffectNodeId>> {
        let mut incoming: HashMap<_, usize> = self.nodes.iter().map(|node| (node.id, 0)).collect();
        let mut outgoing: HashMap<EffectNodeId, Vec<EffectNodeId>> = HashMap::new();
        for (&node, dependencies) in &self.dependencies {
            *incoming.get_mut(&node)? += dependencies.len();
            for &dependency in dependencies {
                outgoing.entry(dependency).or_default().push(node);
            }
        }
        let mut ready: VecDeque<_> = self
            .nodes
            .iter()
            .filter_map(|node| (incoming[&node.id] == 0).then_some(node.id))
            .collect();
        let mut ordered = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_front() {
            ordered.push(id);
            if let Some(dependents) = outgoing.get(&id) {
                for dependent in dependents {
                    let count = incoming.get_mut(dependent)?;
                    *count -= 1;
                    if *count == 0 {
                        ready.push_back(*dependent);
                    }
                }
            }
        }
        (ordered.len() == self.nodes.len()).then_some(ordered)
    }

    /// Largest enabled local radius, or `Global` if any node has global influence.
    pub fn combined_influence(&self, registry: &EffectRegistry) -> EffectInfluence {
        let mut radius_px = 0;
        for node in self.nodes.iter().filter(|node| node.enabled) {
            let Some(descriptor) = registry.get(node.effect) else {
                continue;
            };
            match descriptor.influence {
                EffectInfluence::Local { radius_px: radius } => {
                    radius_px = radius_px.max(radius);
                }
                EffectInfluence::Global => return EffectInfluence::Global,
            }
        }
        EffectInfluence::Local { radius_px }
    }

    fn contains(&self, id: EffectNodeId) -> bool {
        self.nodes.iter().any(|node| node.id == id)
    }
}

/// Reusable scratch-plane allocator shared by every registered paint/effect pass.
#[derive(Clone, Debug, Default)]
pub struct ScratchPlanePool {
    budget_bytes: u64,
    leased_bytes: u64,
    high_water_bytes: u64,
    leases: HashSet<u64>,
    next_lease: u64,
}

impl ScratchPlanePool {
    /// Creates a pool with one aggregate transient memory limit.
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            ..Self::default()
        }
    }

    /// Updates the transient budget after persistent tile allocations change.
    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes.max(self.leased_bytes);
    }

    /// Bytes currently leased by in-flight effect work.
    pub fn leased_bytes(&self) -> u64 {
        self.leased_bytes
    }

    /// Attempts to lease all listed planes for `tile_count` tiles.
    pub fn lease(
        &mut self,
        planes: &[PaintPlaneDescriptor],
        tile_size: u32,
        tile_count: u32,
    ) -> Option<ScratchPlaneLease> {
        let pixels = u64::from(tile_size).pow(2) * u64::from(tile_count);
        let bytes = planes
            .iter()
            .map(|plane| plane.format.bytes_per_pixel() * pixels)
            .sum::<u64>();
        if self.leased_bytes.saturating_add(bytes) > self.budget_bytes {
            return None;
        }
        let id = self.next_lease;
        self.next_lease = self.next_lease.wrapping_add(1);
        self.leases.insert(id);
        self.leased_bytes += bytes;
        self.high_water_bytes = self.high_water_bytes.max(self.leased_bytes);
        Some(ScratchPlaneLease { id, bytes })
    }

    /// Returns a lease to the pool. Duplicate releases are ignored.
    pub fn release(&mut self, lease: ScratchPlaneLease) {
        if self.leases.remove(&lease.id) {
            self.leased_bytes = self.leased_bytes.saturating_sub(lease.bytes);
        }
    }

    /// Peak transient usage since the pool was created.
    pub fn high_water_bytes(&self) -> u64 {
        self.high_water_bytes
    }
}

/// Opaque scratch-plane allocation token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchPlaneLease {
    id: u64,
    bytes: u64,
}

pub(crate) fn pass_through_descriptor() -> EffectDescriptor {
    EffectDescriptor {
        id: PASS_THROUGH_EFFECT_ID,
        implementation_version: 1,
        domain: EffectDomain::LinearDisplayRgba,
        influence: EffectInfluence::Local { radius_px: 0 },
        scratch_planes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PaintPlaneClearValue, PaintPlaneFormat};

    #[test]
    fn graph_rejects_cycles_and_preserves_dependency_order() {
        let mut graph = EffectGraph::default();
        let a = graph.add_node(EffectId(1), 1);
        let b = graph.add_node(EffectId(2), 1);
        graph.add_dependency(b, a).unwrap();
        assert!(graph.add_dependency(a, b).is_err());
        let order = graph.topological_order().unwrap();
        assert!(
            order.iter().position(|id| *id == a).unwrap()
                < order.iter().position(|id| *id == b).unwrap()
        );
    }

    #[test]
    fn scratch_pool_accounts_for_every_declared_plane() {
        let planes = [PaintPlaneDescriptor {
            name: "mask",
            semantic: "test scratch",
            format: PaintPlaneFormat::R32Float,
            clear: PaintPlaneClearValue::Zero,
        }];
        let expected = 64 * 64 * 4;
        let mut pool = ScratchPlanePool::with_budget(expected);
        let lease = pool.lease(&planes, 64, 1).unwrap();
        assert!(pool.lease(&planes, 64, 1).is_none());
        pool.release(lease);
        assert!(pool.lease(&planes, 64, 1).is_some());
        assert_eq!(pool.high_water_bytes(), expected);
    }

    #[test]
    fn nonzero_radius_extension_changes_common_graph_influence() {
        let blur_id = EffectId(41);
        let mut registry = EffectRegistry::default();
        registry.register(pass_through_descriptor());
        registry.register(EffectDescriptor {
            id: blur_id,
            implementation_version: 7,
            domain: EffectDomain::LinearDisplayRgba,
            influence: EffectInfluence::Local { radius_px: 19 },
            scratch_planes: vec![PaintPlaneDescriptor {
                name: "ping",
                semantic: "mock blur scratch",
                format: PaintPlaneFormat::Rgba16Float,
                clear: PaintPlaneClearValue::Zero,
            }],
        });
        let mut graph = EffectGraph::default();
        graph.add_node(blur_id, 7);
        assert_eq!(
            graph.combined_influence(&registry),
            EffectInfluence::Local { radius_px: 19 }
        );
    }
}
