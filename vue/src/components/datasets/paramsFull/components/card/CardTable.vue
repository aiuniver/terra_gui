<template>
  <div class="csv-table">
    <div class="grouped__actions">
      <t-button v-if="selected_cols.length" @click.native="grouping">Связать</t-button>
      <t-field inline v-if="selected_cols.length">
        <t-select-new @change="selectGroup = $event.value" :list="list_inp_out" small />
      </t-field>
    </div>
    <div class="table__data">
      <div class="table__col">
        <div class="table__row"></div>
        <div class="table__row">0</div>
        <div class="table__row">2</div>
        <div class="table__row">4</div>
        <div class="table__row">6</div>
        <div class="table__row">8</div>
      </div>
      <div class="selected__cols"></div>

      <div id="grouped__cols">
        <div class="grouped__col" v-for="{ value: group } in list_inp_out" :key="'k_' + group" :data-group="group">
          <div class="grouped__col-content" :data-group="group">
            <small class="grouped__cols-headline">{{ group }}</small>
          </div>
        </div>
      </div>
      <div
        class="table__col"
        v-for="(row, index) in arr"
        :key="'row_' + index"
        @click="select(index)"
        :data-index="index"
      >
        <div class="table__row" v-for="(item, i) in row" :key="'item_' + i">{{ item }}</div>
      </div>
    </div>

    <div class="table__footer">
      <span>Список файлов</span>
      <div class="table__footer--btn" @click="show = true">
        <i class="t-icon icon-file-dot"></i>
      </div>
      <div v-show="show" class="table__dropdown">
        <div
          v-for="({ icon, event }, i) of items"
          :key="'icon' + i"
          class="table__dropdown--item"
          @click="$emit('event', { label, event }), (show = false)"
        >
          <i :class="['t-icon', icon]"></i>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CardTable',
  props: {
    label: String,
    type: String,
    id: Number,
    cover: String,
    table: Array,
  },
  data: () => ({
    list_inp_out: [
      { label: 'outpu1', value: 'outpu1' },
      { label: 'input1', value: 'input1' },
      { label: 'outpu2', value: 'outpu2' },
    ],
    selectGroup: null,
    table_test: [],
    selected_cols: [],
    grouped_cols: [],
    show: false,
    items: [{ icon: 'icon-deploy-remove', event: 'remove' }],
  }),
  computed: {
    arr() {
      const newarr = [];
      this.table.forEach((el, index) => {
        el.forEach((elm, i) => {
          if (!newarr[i]) {
            newarr[i] = [];
          }
          newarr[i][index] = elm;
        });
      });
      console.log(newarr);
      return newarr;
    },
  },
  created() {
    console.log(this.table);
    this.list_inp_out.forEach(el => {
      this.grouped_cols.push({
        groupName: el.value,
        selectedCols: [],
      });
    });
  },
  methods: {
    grouping() {
      if (this.selected_cols.length > 0 && this.selectGroup) {
        this.grouped_cols = this.grouped_cols.map(el => {
          if (el.groupName === this.selectGroup)
            return { groupName: this.selectGroup, selectedCols: [...new Set(this.selected_cols)] };
          return el;
        });
        const group = document.querySelector(`.grouped__col[data-group="${this.selectGroup}"] > .grouped__col-content`);

        this.grouped_cols
          .find(el => el.groupName === this.selectGroup)
          .selectedCols.forEach(el => {
            group.appendChild(document.querySelector(`.table__col[data-index='${el}']`));
          });
        group.style.display = 'flex';

        for (let el of document.querySelectorAll(`.grouped__col-content`))
          if (el.length === 0) el.style.display = 'none';

        document.querySelector('.selected__cols').style.display = 'none';
        this.selected_cols = [];
        this.selectGroup = '';
      }
    },
    compare(a, b) {
      if (a.dataset.index < b.dataset.index) {
        return -1;
      }
      if (a.dataset.index > b.dataset.index) {
        return 1;
      }
      return 0;
    },
    sortOnDataIndex(el) {
      let arr = [],
        i = el.children.length;
      while (i--) {
        arr[i] = el.children[i];
        el.children[i].remove();
      }
      arr.sort(this.compare);
      i = 0;
      while (arr[i]) {
        el.appendChild(arr[i]);
        ++i;
      }
    },
    select(index) {
      event.preventDefault();
      const selected = document.querySelector('.selected__cols');
      const table = document.querySelector('.table__data');
      const col = document.querySelector(`.table__col[data-index="${index}"]`);
      if (!this.selected_cols.includes(index)) {
        this.selected_cols.push(index);
        if (col.parentElement.hasAttribute('data-group') && col.parentElement.childNodes.length === 2)
          col.parentElement.style.display = 'none';
        selected.appendChild(col);
      } else {
        this.selected_cols.splice(this.selected_cols.indexOf(index), 1);
        const checkGroup = this.grouped_cols.find(el => el.selectedCols.includes(index));
        if (checkGroup) {
          document.querySelector(`.grouped__col-content[data-group="${checkGroup.groupName}"]`).appendChild(col);
          document.querySelector(`.grouped__col-content[data-group="${checkGroup.groupName}"]`).style.display = 'flex';
        } else {
          this.grouped_cols = this.grouped_cols.map(el => {
            if (el.selectedCols.includes(index)) el.selectedCols.splice(el.selectedCols.indexOf(index), 1);
            if (el.selectedCols.length === 0)
              document.querySelector(`.grouped__col-content[data-group="${el.groupName}"]`).style.display = 'none';

            return {
              groupName: el.groupName,
              selectedCols: el.selectedCols,
            };
          });
          table.appendChild(col);
        }
      }
      this.sortOnDataIndex(table);
      if (this.selected_cols.length > 0) {
        selected.style.display = 'flex';
      } else {
        selected.style.display = 'none';
      }
    },
  },
};
</script>

<style lang="scss" scoped>
#grouped__cols {
  display: flex;
}
.grouped__col-content {
  border: 1px solid #89d764;
  border-radius: 4px;
  play: flex;
  display: none;
  position: relative;
}
.grouped__cols-headline {
  position: absolute;
  top: -18px;
}

.selected__cols {
  display: flex;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.selected {
  border: 1px solid #ccc;
  border-radius: 4px;
  color: #fff;

  &:nth-child(1) {
    border-radius: 6px 0 0 6px;
  }
  &:last-child {
    border-radius: 0 6px 6px 0;
  }
}

.csv-table {
  font-size: 0.75rem;
  border-collapse: collapse;
  border: 1px solid #6c7883;
  border-radius: 8px;
  padding: 23px 0 2px 0;
  display: flex;
  height: 152px;
  flex-direction: column;

  .table__data {
    display: flex;
  }

  .table__col {
    display: flex;
    flex-direction: column;
  }
  .table__row {
    height: 17px;
    padding: 0 8px;
    &:nth-child(even) {
      background: #242f3d;
    }
    &:first-child {
      font-weight: bold;
      height: 19px;
    }
  }

  .table__footer {
    height: 24px;
    width: 100%;
    padding: 3px 8px;
    position: relative;
    display: flex;
    justify-content: space-between;
    align-items: center;

    &--label {
      bottom: 0;
      border-radius: 0 0 3px 3px;
      padding: 4px 2px 2px 6px;
      text-overflow: ellipsis;
      overflow: hidden;
    }
    &--btn {
      padding: 0 6px 0 0;
      cursor: pointer;
      i {
        width: 16px;
      }
      &:hover {
      }
    }
  }
  .table__dropdown {
    position: absolute;
    background-color: #2b5278;
    border-radius: 4px;
    right: 3px;
    bottom: 3px;
    z-index: 100;
    &--item {
      position: relative;
      width: 32px;
      height: 32px;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      &:hover {
        opacity: 0.7;
      }
      i {
        width: 14px;
      }
    }
  }
}
</style>
