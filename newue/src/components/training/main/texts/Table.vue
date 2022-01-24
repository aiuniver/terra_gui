<template>
  <div class="t-table">
    <scrollbar :ops="ops">
      <table v-if="isTable">
        <thead>
          <tr class="outputs_heads">
            <th rowspan="2">Эпоха</th>
            <th rowspan="2">
              Время
              <br />
              (сек.)
            </th>
            <template v-for="(output, key) of outputs">
              <th v-show="isShow(key)" :key="key" colspan="100%">
                {{ key }}
              </th>
            </template>
          </tr>
          <tr class="callbacks_heads">
            <template v-for="(output, keyO) of outputs">
              <template v-for="(item, keyI) of output">
                <template v-for="(th, key) of item">
                  <th v-show="isShow(keyO, keyI)" :key="keyI + key + keyO">{{ key }}</th>
                </template>
              </template>
            </template>
          </tr>
        </thead>
        <tbody>
          <tr v-for="({ time, data }, key, i) of data" :key="'epoch_' + i">
            <td class="epoch_num">{{ key }}</td>
            <td>{{ time | ceil }}</td>
            <template v-for="(output, keyO) of data">
              <template v-for="(metric, keyM) of output">
                <template v-for="(item, keyI) of metric">
                  <td v-show="isShow(keyO, keyM)" class="value" :key="keyO + 't' + keyM + 'r' + keyI">
                    <span>{{ item }}</span>
                  </td>
                </template>
              </template>
            </template>
          </tr>
        </tbody>
        <tfoot>
          <tr>
            <th colspan="100%">{{ 'summary' }}</th>
          </tr>
        </tfoot>
      </table>
    </scrollbar>
  </div>
</template>

<script>
export default {
  name: 't-table',
  props: {
    data: {
      type: Object,
      default: () => {},
    },
    settings: {
      type: Object,
      default: () => {},
    },
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
  }),
  mounted() {
    console.log(this.settings);
  },
  computed: {
    isTable() {
      return !!this.data?.[1];
    },
    outputs() {
      return this.data?.[1]?.data || {};
    },
  },
  filters: {
    int(val) {
      return ~~val;
    },
    ceil(val) {
      return Math.ceil(val);
    },
    drob(val) {
      return (val % 1).toFixed(9).slice(2);
    },
  },
  methods: {
    isShow(layer, metrics) {
      // console.log(layer, metrics)
      if (layer && metrics) {
        return this.settings?.[layer]?.[metrics] ?? true;
      } else {
        return (this.settings?.[layer]?.loss ?? true) || (this.settings?.[layer]?.metrics ?? true);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-table {
  width: 100%;
  position: relative;
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
}
</style>