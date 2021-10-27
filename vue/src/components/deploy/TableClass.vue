<template>
  <div class="content">
    <table class="table">
      <tr class="table__title-row stick">
        <td>
          <button class="table__reload-all" @click="ReloadAll">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
            <span>Перезагрузить все</span>
          </button>
        </td>
        <td>Предсказанные данные</td>
        <td>{{ columns[columns.length-1] }}</td>
        <td v-for="(name, i) of columns.slice(0, columns.length-1)" :key="'col_' + i">{{ name }}</td>
      </tr>
      <tr v-for="({ source, data, actual }, index) of data" :key="'row_' + index" class="fixed">
        <td class="table__td-reload">
          <button class="td-reload__btn-reload" @click="ReloadRow(index)">
            <i :class="['t-icon', 'icon-deploy-reload']" :title="'reload'"></i>
          </button>
        </td>
        <td class="table__result-data table__result-data--left">
          <div v-for="(item, i) of data" :key="'esul_' + i">
            <span>{{ item[0] }}</span>
            -
            <span>{{ item[1] }}</span>
          </div>
        </td>
        <td><span class="table__result-data--actual">{{ actual }}</span></td>
        <td v-for="(data, i) of source" :key="'data_' + index + i">{{ data }}</td>
      </tr>
    </table>
  </div>
</template>

<script>
export default {
  name: 'Table',
  props: {
    data: Array,
    source: Object,
    columns: Object,
  },
  data: () => ({}),
  // computed: {
  //   columns() {
  //     return this.extra?.columns ?? [];
  //   },
  // },
  methods: {
    ReloadRow(index) {
      console.log('RELOAD_ROW');
      this.$emit('reload', [index.toString()]);
    },
    ReloadAll() {
      this.$emit('reloadAll');
    },
  },
  mounted() {
    console.log(this.data)
    console.log(this.columns)
  },
};
</script>

<style scoped lang="scss">
.content {
  height: 770px;
  // width: 1200px;
}
.table {
  width: 100%;
  background: #242f3d;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  position: sticky;
  .stick {
    position: sticky;
    top: 5px;
    background: #242f3d;
  }
  .fixed {
    overflow: hidden;
  }
  tr:nth-child(even) {
    background: #17212b;
  }
  td {
        height: 50px;
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
  &--left {
    text-align: left !important;
  }
  &--actual{
    color: #0bbc49;
  }
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