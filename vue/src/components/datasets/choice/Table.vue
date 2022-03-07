<template>
  <table>
    <thead>
      <tr>
        <th v-for="(header, idx) in headers" :key="'table_dataset_th' + idx" @click="handleSort(header.idx)">
          <span>{{ header.title }}</span>
        </th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(dataset, idx) in data" :key="'table_dataset_tr' + idx" @click="$emit('choice', dataset)" :class="{ selected: isActive(dataset.id) }">
        <td v-for="({ value }, idx) in headers" :key="'table_dataset_td' + idx">
          <span v-if="value !== 'group'">{{ dataset[value] }}</span>
          <span v-else>{{ $store.getters['datasets/getGroups'][dataset[value]] }}</span>
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
      // {
      //   title: 'Размер',
      //   value: 'size',
      //   idx: 1,
      // },
      {
        title: 'Группа',
        value: 'group',
        idx: 2,
      },
      // {
      //   title: 'Последнее использование',
      //   value: 'date',
      //   idx: 3,
      // },
      // {
      //   title: 'Создание',
      //   value: 'alias',
      //   idx: 4,
      // },
    ],
    sortId: 0,
    sortReverse: false,
  }),
  methods: {
    handleSort(idx) {
      if (this.sortId === idx) {
        if (this.sortReverse) this.sortReverse = false;
        else this.sortReverse = true;
      } else this.sortReverse = false;

      this.sortId = idx;
    },
    isActive(id) {
      const { alias, group } = this.$store.getters['projects/getProject'].dataset || {}
      return `${group}_${alias}` === id
    },
  },
  computed: {
    // datasets() {
    //   const items = this.data;
    //   if (this.sortId === 0)
    //     return this.sortReverse
    //       ? items.sort((a, b) => b.name.localeCompare(a.name))
    //       : items.sort((a, b) => a.name.localeCompare(b.name));
    //   return items;
    // },
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
@import "@/assets/scss/variables/default.scss";
@mixin cell {
  // width: 25%;
  height: 35px;
  padding: 0 10px;
  font-weight: 400;
  text-align: left;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  max-width: 100px;
  &:nth-child(2) {
    // text-align: right;
  }
  @content;
}

table {
  font-size: 14px;
  width: 100%;
}

thead {
  background-color: #17212b;
  color: $color-gray;
  position: sticky;
  top: 0;
  box-shadow: 0 0 3px black;
  th:not(:first-child) {
    width: 15%;
  }
  th:last-child {
    width: 20%;
  }
}

tbody {
  tr {
    cursor: pointer;
    &:hover {
      background: #1E2734;
      color: #65B9F4;
    }
  }
}

tr {
  &.selected {
    color: #3eba31;
  }
  th,td {
    @include cell
  }

  td {
    height: 55px;
    &:not(:first-child) {
      color: $color-gray-blue;
    }
  }

}
</style>