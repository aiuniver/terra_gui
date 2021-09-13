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
      <tr v-for="({data:{output_1:{accuracy, loss, val_accuracy, val_loss}}, time, number}, i) of epochs" :key="'epoch_' + i">
        <td class="epoch_num">{{ number }}</td>
        <td>{{ time }}</td>
        <td class="value"><span>{{ accuracy | int }}</span><i>.</i>{{ accuracy | drob }}</td>
        <td class="value"><span>{{ loss | int }}</span><i>.</i>{{ loss | drob }}</td>
        <td class="value"><span>{{ val_accuracy | int }}</span><i>.</i>{{ val_accuracy | drob }}</td>
        <td class="value"><span>{{ val_loss | int }}</span><i>.</i>{{ val_loss | drob }}</td>

      </tr>
    </tbody>
    <tfoot>
      <tr>
        <th colspan="6">{{ summary }}</th>
      </tr>
    </tfoot>
  </table>
</template>

<script>
export default {
  name: "TTable",
  props: {
    summary: {
      type: String,
      default: ''
    },
    epochs: {
      type: Array,
      default: () => [],
    },
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