<template>
  <table class="datasets-table">
    <thead>
      <tr>
        <th v-for="(header, idx) in headers" :key="'table_dataset_th'+idx" @click="handleSort(header.idx)">
          <span>{{ header.title }}</span>
        </th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(dataset, idx) in datasets" :key="'table_dataset_tr' + idx">
          <td v-for="({value}, idx) in headers" :key="'table_dataset_td' + idx">
              <span>{{ dataset[value] }}</span>
          </td>
      </tr>
    </tbody>
  </table>
</template>

<script>
export default {
  name: 'Table',
  props: {
    data: {
      type: Array,
      default: () => [],
    },
    selectedType: {
      type: [String, Number],
      default: 0,
    },
  },
  data: () => ({
    list: [
      {
        title: 'Название',
        value: 'name',
        idx: 0,
      },
      {
        title: 'Размер',
        value: 'size',
        idx: 1,
      },
      {
        title: 'Автор',
        value: 'group',
        idx: 2,
      },
      {
        title: 'Последнее использование',
        value: 'date',
        idx: 3,
      },
      {
        title: 'Создание',
        value: 'alias',
        idx: 4,
      },
    ],
    sortId: 0,
    sortReverse: false,
  }),
  methods: {
    handleSort(idx) {
      if(this.sortId === idx){
          if(this.sortReverse) this.sortReverse = false
          else this.sortReverse = true
      }else this.sortReverse = false

      this.sortId = idx;
    },
  },
  computed: {
    datasets() {
      const items = this.data;
      if (this.sortId === 0)
        return this.sortReverse
          ? items.sort((a, b) => b.name.localeCompare(a.name))
          : items.sort((a, b) => a.name.localeCompare(b.name));
      return items;
    },
    headers() {
      const arr = [...this.list];
      if (this.selectedType === 1) {
        arr.splice(2, 1);
        return arr;
      }
      return arr.slice(0, 4);
    },
  },
};
</script>

<style lang="scss" scoped>
.datasets-table {
  font-size: 14px;
  font-weight: 400;
  width: 100%;
  &__wrapper {
    width: calc(100% - 150px);
    position: relative;
  }
  tr > *:nth-child(2) {
    text-align: right;
  }
  thead {
    background-color: #17212b;
    color: #6c7883;
    position: sticky;
    top: 0;
    i {
      color: #65b9f4;
      font-size: 20px;
      vertical-align: middle;
    }
    tr {
      height: 35px;
    }
    th {
      font-weight: inherit;
      padding: 0 50px;
      min-width: 150px;
      user-select: none;
      &:first-child {
        padding: 15px 10px;
      }
      * {
        vertical-align: middle;
        cursor: pointer;
      }
    }
  }
  tbody {
    tr {
      height: 55px;
      cursor: pointer;
      &:hover {
        background-color: #0e1621;
      }
    }
    td {
      color: #f2f5fa;
      padding: 15px 50px;
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
      max-width: 450px;
      &:first-child {
        padding: 15px 10px;
      }
      i {
        font-size: 19px;
        color: #6c7883;
        margin-right: 15px;
      }
      * {
        vertical-align: middle;
      }
    }
  }
}
</style>