<template>
  <table class="datasets-table">
    <thead>
      <tr>
        <th v-for="(header, idx) in headers" :key="idx" @click="handleSort(header.idx)">
          <span>{{ header.title }}</span>
        </th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(dataset, idx) in datasets" :key="'table' + idx">
        <td>
          <span>{{ dataset.name }}</span>
        </td>
        <td>
          {{ dataset.size ? `${dataset.size.short.toFixed(2)} ${dataset.size.unit}` : 'Предустановленный' }}
        </td>
        <td></td>
        <td>{{ dataset.date ? dataset.date.toLocaleString() : '' }}</td>
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
        idx: 0,
      },
      {
        title: 'Размер',
        idx: 1,
      },
      {
        title: 'Автор',
        idx: 2,
      },
      {
        title: 'Последнее использование',
        idx: 3,
      },
      {
        title: 'Создание',
        idx: 4,
      },
    ],
    sortId: 0,
    sortReverse: false,
  }),
  methods: {
    handleSort(idx) {
      this.sortReverse = this.sortId === idx ? true : false;
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
      if (this.sortId === 1)
        return this.sortReverse
          ? items.sort((a, b) => b.size.value - a.size.value)
          : items.sort((a, b) => a.size.value - b.size.value);
      if (this.sortId === 2)
        return this.sortReverse
          ? items.sort((a, b) => a.name.localeCompare(b.name))
          : items.sort((a, b) => b.name.localeCompare(a.name));
      if (this.sortId === 3)
        return this.sortReverse ? items.sort((a, b) => a.date - b.date) : items.sort((a, b) => b.date - a.date);
      if (this.sortId === 4)
        return this.sortReverse ? items.sort((a, b) => a.date - b.date) : items.sort((a, b) => b.date - a.date);
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