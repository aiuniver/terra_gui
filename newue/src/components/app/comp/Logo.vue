<template>
  <div class="logo">
    <div class="logo__logo"></div>
    <i class="ci-icon cl-circle_chevron_right" />
    <ul class="dropdown">
      <li v-for="route of routs" class="dropdown__item" :key="route.title">
        {{ route.title }}
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'logo',
  props: {
    // routs: {
    //   type: Array,
    //   default: () => ([])
    // }
  },
  data: () => ({
    routs: [
      {
        title: 'Датасет',
        list: [{ label: 'Создание' }, { label: 'Создание' }, { label: 'Создание' }],
      },
      {
        title: 'Каскады',
        list: [{ label: 'Создание' }, { label: 'Создание' }, { label: 'Создание' }],
      },
    ],
  }),
};
</script>

<style lang="scss" scoped>
.logo {
  position: relative;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  &__logo {
    cursor: pointer;
    width: 28px;
    height: 28px;
    background-position: center;
    background-repeat: no-repeat;
    background-size: contain;
    background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIxLjQxNzUgOC4wNzA2QzIyLjIzMjggOC4wNzA2IDIyLjg5OTkgNy40MDM1NCAyMi44OTk5IDYuNTg4MjVDMjIuODk5OSA1Ljc3Mjk2IDIyLjIzMjggNS4xMDU5IDIxLjQxNzUgNS4xMDU5SDYuNTk0MDJDNS43Nzg3MiA1LjEwNTkgNS4xMTE2NiA1Ljc3Mjk2IDUuMTExNjYgNi41ODgyNUM1LjExMTY2IDcuNDAzNTQgNS43Nzg3MiA4LjA3MDYgNi41OTQwMiA4LjA3MDZIMjEuNDE3NVpNMTIuNTIzNCAyMS40MTE4QzEyLjUyMzQgMjIuMjI3MSAxMy4xOTA1IDIyLjg5NDEgMTQuMDA1OCAyMi44OTQxQzE0LjgyMTEgMjIuODk0MSAxNS40ODgxIDIyLjIyNzEgMTUuNDg4MSAyMS40MTE4VjEyLjc2NDdDMTUuNDg4MSAxMS45NDk0IDE0LjgyMTEgMTEuMjgyNCAxNC4wMDU4IDExLjI4MjRDMTMuMTkwNSAxMS4yODI0IDEyLjUyMzQgMTEuOTQ5NCAxMi41MjM0IDEyLjc2NDdWMjEuNDExOFoiIGZpbGw9IiM2NUI5RjQiLz4KPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0yNy4xNzY1IDAuODIzNTI5SDAuODIzNTI5VjI3LjE3NjVIMjcuMTc2NVYwLjgyMzUyOVpNMCAwVjI4SDI4VjBIMFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==');
  }
}
</style>