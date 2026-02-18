from __future__ import annotations


def build_world(
    *,
    isc_cls,
    user_agent_cls,
    engine_cls,
    num_genres: int,
    content_g_params,
    num_contents: int,
    num_agents: int,
    device,
    configure_user_agent_hooks,
):
    isc = isc_cls(num_genres, content_g_params, num_contents)
    configure_user_agent_hooks()
    agents = [user_agent_cls(i) for i in range(num_agents)]

    engine = engine_cls(num_agents, num_contents, num_genres, device)
    engine.load_agents(agents)
    engine.load_contents(isc.pool)
    return isc, agents, engine

