<template>
  <table>
    <thead>
      <tr class="outputs_heads">
        <th rowspan="2">Эпоха</th>
        <th rowspan="2">Время<br />(сек.)</th>
        <th colspan="4">output_1</th>
      </tr>
      <tr class="callbacks_heads">
        <th>accuracy</th>
        <th>val_accuracy</th>
        <th>loss</th>
        <th>val_loss</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="({time, data}, key, i) of data" :key="'epoch_' + i">
        <td class="epoch_num">{{ key }}</td>
        <td>{{ time | int }}</td>
        <template v-for="item, key, i of data">
          <td class="value" :key="key + '_tr_1' + i"><span>{{ item.loss.loss | drob }}</span><i>.</i>{{ '' }}</td>
          <td class="value" :key="key + '_tr_2' + i"><span>{{ item.loss.val_loss | drob }}</span><i>.</i>{{ '' }}</td>
          <td class="value" :key="key + '_tr_3' + i"><span>{{ item.metrics.AUC | drob}}</span><i>.</i>{{ '' }}</td>
          <td class="value" :key="key + '_tr_4' + i"><span>{{ item.metrics.val_AUC | drob }}</span><i>.</i>{{ '' }}</td>
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
  name: "TTable",
  props: {
    data: {
      type: Object,
      default: () => {}
    }    
  },
  filters: {
    int(val) {
      return ~~val
    },
    drob(val) {
      return (val%1).toFixed(3).slice(2)
    }
  }
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