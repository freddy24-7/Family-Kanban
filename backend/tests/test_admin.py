from tests.helpers import create_household, make_superuser, register_and_login


async def test_admin_endpoints_require_superuser(client, outbox):
    user = await register_and_login(client, "user@example.com")
    household = await create_household(client, user)
    assert (await client.get("/admin/households", headers=user)).status_code == 403
    response = await client.patch(
        f"/admin/households/{household}", json={"training_eligible": True}, headers=user
    )
    assert response.status_code == 403


async def test_admin_marks_household_training_eligible(client, outbox):
    user = await register_and_login(client, "fam@example.com")
    household = await create_household(client, user)
    admin = await register_and_login(client, "root@example.com")
    await make_superuser("root@example.com")

    assert (await client.get(f"/households/{household}", headers=user)).json()[
        "training_eligible"
    ] is False
    response = await client.patch(
        f"/admin/households/{household}", json={"training_eligible": True}, headers=admin
    )
    assert response.json()["training_eligible"] is True
    assert len((await client.get("/admin/households", headers=admin)).json()) == 1


async def test_admin_can_access_any_household(client, outbox):
    user = await register_and_login(client, "fam2@example.com")
    household = await create_household(client, user)
    admin = await register_and_login(client, "root2@example.com")
    await make_superuser("root2@example.com")
    assert (await client.get(f"/households/{household}/topics", headers=admin)).status_code == 200
