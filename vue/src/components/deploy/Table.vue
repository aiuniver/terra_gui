<template>
  <div class="content">
    <table class="table">
      <tr class="table__title-row">
        <td>
          <button class="table__reload-all" @click="ReloadAll">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
            <span>Перезагрузить все</span>
          </button>
        </td>
        <td>Предсказанные данные</td>
        <td v-for="(name, i) of columns" :key="'col_' + i">{{ name }}</td>
      </tr>
      <tr v-for="({ preset, label }, index) of data" :key="'row_' + index">
        <td class="table__td-reload">
          <button class="td-reload__btn-reload" @click="ReloadRow(index)">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
          </button>
        </td>
        <td class="table__result-data">{{ label }}</td>
        <td v-for="(data, i) of preset" :key="'data_' + index + i">
          <TableText v-bind="{value: data}" :style="{width: '100%'}"/>
<!--          {{ data }}-->
        </td>
        <td class="table__result-data">{{ label }}</td>
      </tr>
    </table>
  </div>
</template>

<script>
export default {
  name: 'Table',
  components: {
    TableText: () => import('../training/main/prediction/components/TableText')
  },
  props: {
    data: Array,
    columns: Array,
  },
  // computed: {
  //   columns() {
  //     return this.extra?.columns ?? [];
  //   },
  // },
  methods: {
    ReloadRow(index) {
      this.$emit('reload', [index.toString()]);
    },
    ReloadAll() {
      this.$emit('reloadAll');
    },
  },
  mounted() {
    console.log(12312312);
    console.log(this.data);
    console.log(this.columns);
  },
};
</script>

<style scoped lang="scss">
.table {
  width: 100%;
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  tr:nth-child(even) {
    background: #17212b;
  }
  td {
    padding: 5px;
    text-align: center;
    &:nth-child(1) {
      padding-left: 74px;
    }
  }
}
.table__title-row {
  color: #ffffff;
  font-weight: bold;
  font-size: 14px;
  line-height: 24px;
  td {
    text-align: left;
    text-align: center;
    &:nth-child(1) {
      padding-left: 8px;
    }
  }
}
.table__result-data {
  width: 200px;
  color: #65b9f4;
}
.table__reload-all {
  display: flex;
  width: 174px;
  padding: 8px 10px 10px 10px;
  justify-content: center;
  align-items: center;
  i {
    width: 16px;
  }
  span {
    font-weight: normal;
    font-size: 14px;
    line-height: 24px;
    padding-left: 8px;
  }
}
.table__td-reload {
  text-align: center;
}
.td-reload__btn-reload {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 32px;
  height: 32px;
  padding-bottom: 2px;
  i {
    width: 16px;
  }
}
</style>