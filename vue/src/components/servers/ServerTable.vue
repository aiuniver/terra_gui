<template>
  <table class="server-table">
    <thead>
      <tr>
        <th>Доменное имя</th>
        <th>IP адрес</th>
        <th>Имя пользователя</th>
        <th>SSH порт</th>
        <th>HTTP порт</th>
        <th>HTTPS порт</th>
        <th>Инструкция</th>
        <th>Состояние</th>
        <th style="min-width: 144px"></th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="server in servers" :key="server.id">
        <td>{{ server.domain_name }}</td>
        <td>{{ server.ip_address }}</td>
        <td>{{ server.user }}</td>
        <td>{{ server.port_ssh }}</td>
        <td>{{ server.port_http }}</td>
        <td>{{ server.port_https }}</td>
        <td class="clickable"><span @click="instruction(server.id)">Открыть</span></td>
        <td v-if="server.state.error">
          <span>{{ server.state.error }}</span>
          <i class="ci-icon ci-info_circle_outline"></i>
        </td>
        <td v-else>{{ server.state.value }}</td>
        <td class="clickable" @click="setup(server.id)">
          <i :class="['ci-icon', getIcon(server.state.name)]"></i>
          <span>{{ getAction(server.state.name) }}</span>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<script>
export default {
  name: 'ServerTable',
  props: {
    servers: Array
  },
  methods: {
    instruction(id) {
      this.$emit('instruction', id)
    },
    setup(id) {
      this.$store.dispatch('servers/setup', { id })
    },
    getIcon(state) {
      if (state === 'ready') return 'ci-redo'
      if (state === 'idle') return ''
      return 'ci-play_arrow'
    },
    getAction(state) {
      if (state === 'ready') return 'Обновить запуск'
      if (state === 'idle') return ''
      return 'Запустить'
    }
  }
};
</script>

<style lang="scss" scoped>
.server-table {
  width: 100%;
  thead {
    color: #6c7883;
    font-size: 14px;
    height: 35px;
    th {
      font-weight: 400;
    }
  }
  tbody {
    font-size: 14px;
    tr {
      height: 55px;
      &:hover {
        background: #0e1621;
      }
    }
  }
  td,
  th {
    padding-right: 10px;
    &:first-child {
      padding-left: 10px;
    }
  }
  .clickable {
    color: #65b9f4;
    > * {cursor: pointer;}
    i {
      font-size: 16px;
      margin-right: 5px;
    }
    span {
      position: relative;
      vertical-align: text-bottom;
      &:hover::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 1px;
        background: #65b9f4;
      }
    }
  }
  .error {
    color: #ca5035;
    i {
      color: #65b9f4;
      font-size: 16px;
      vertical-align: middle;
    }
  }
}
</style>