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
        <td :class="[`${server.state.name}`]">{{ server.state.value }}</td>
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
    servers: [Array, Object]
  },
  methods: {
    instruction(id) {
      this.$emit('instruction', id)
    },
    setup(id) {
      this.$store.dispatch('servers/setup', { id })
    },
    getIcon(state) {
      if (state === 'ready') return ''
      // if (state === 'idle') return 'ci-play_arrow'
      if (state === 'waiting') return ''
      return 'ci-play_arrow'
    },
    getAction(state) {
      if (state === 'ready') return ''
      // if (state === 'idle') return 'Установить'
      if (state === 'waiting') return ''
      return 'Установить'
    }
  }
};
</script>

<style lang="scss" scoped>
.server-table {
  width: 100%;
  .error {
    color: #f44336;
  }
  .idle {
    color: #9e9e9e;
  }
  .ready {
    color: #8bc34a;
  }
  .waiting {
    color: #ff9800;
  }
  thead {
    color: #6c7883;
    font-size: 12px;
    th {
      font-weight: 400;
    }
  }
  tbody {
    font-size: 14px;
    tr {
      &:hover {
        background: #0e1621;
      }
    }
  }
  td,
  th {
    padding: 10px 20px 10px 0;
    white-space: nowrap;
    line-height: 1.25;
    &:first-child {
      padding-left: 20px;
    }
  }
  .clickable {
    color: #65b9f4;
    > * {cursor: pointer;}
    i {
      font-size: 16px;
      margin-right: 5px;
      vertical-align: middle;
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
    i {
      color: #65b9f4;
      font-size: 16px;
      vertical-align: middle;
    }
  }
}
</style>