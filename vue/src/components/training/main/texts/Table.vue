<template>
  <table>
    <thead>
      <tr class="outputs_heads">
        <th rowspan="2">Эпоха</th>
        <th rowspan="2">
          Время
          <br />
          (сек.)
        </th>
        <th v-for="(output, key) of outputs" :key="key" colspan="4">
          {{ key }}
        </th>
      </tr>
      <tr class="callbacks_heads">
        <template v-for="(output, keyO) of outputs">
          <template v-for="(item, keyI) of output">
            <th v-for="(th, key) of item" :key="keyI + key + keyO">{{ key }}</th>
          </template>
        </template>
      </tr>
    </thead>
    <tbody>
      <tr v-for="({ time, data }, key, i) of data" :key="'epoch_' + i">
        <td class="epoch_num">{{ key }}</td>
        <td>{{ time | int }}</td>
        <template v-for="(output, keyO) of data">
          <template v-for="(metric, keyM) of output">
            <td v-for="(item, keyI) of metric" class="value" :key="keyO + 't' + keyM + 'r' + keyI">
              <span>{{ item | int }}</span>
              <i>.</i>
              {{ item | drob }}
            </td>
          </template>
        </template>
      </tr>
    </tbody>
    <tfoot>
      <tr>
        <th colspan="6">{{ 'summary' }}</th>
      </tr>
    </tfoot>
  </table>
</template>

<script>
export default {
  name: 't-table',
  props: {
    data: {
      type: Object,
      default: () => {},
    },
  },
  computed: {
    outputs() {
      return this.data?.[1]?.data || {};
    },
  },
  filters: {
    int(val) {
      return ~~val;
    },
    drob(val) {
      return (val % 1).toFixed(9).slice(2);
    },
  },
};
</script>

<style lang="scss" scoped>
table {
  user-select: none;
  border-collapse: collapse;
  border-spacing: 1px;
  border: 1px solid #242f3d;
  box-sizing: border-box;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
  border-radius: 4px;
  th {
    background: #242f3d;
    line-height: 16px;
    padding: 5px 10px;
    color: #a7bed3;
    font-size: 12px;
    text-align: center;
    border: 1px solid #0e1621;
  }
  td {
    line-height: 16px;
    padding: 5px 10px;
    color: #fff;
    font-size: 12px;
    text-align: center;
    border: 1px solid #0e1621;
    &.value {
      text-align: center;
    }
  }
}
</style>