<template>
  <div :class="['dataset-card-item', { active: dataset.active && !loaded, selected: loaded }]" :title="dataset.name">
    <div
      :class="['dataset-card', { disabled: !dataset.training_available }]"
      @click.stop="$emit('click', dataset, cardIndex)"
    >
      <div class="card-title">{{ dataset.name }}</div>
      <div class="card-body" @click="click">
        <div v-if="dataset.tags.length <= 4">
          <div v-for="({ name }, key) of dataset.tags" :key="`tag_${key}`" class="card-tag">
            {{ name }}
          </div>
        </div>
        <scrollbar v-else :ops="{ bar: { background: '#17212b' } }">
          <div v-for="({ name }, key) of dataset.tags" :key="`tag_${key}`" class="card-tag">
            {{ name }}
          </div>
        </scrollbar>
      </div>

      <div :class="'card-extra ' + (dataset.size ? 'is-custom' : '')">
        <div class="wrapper">
          <span>
            {{ dataset.size ? dataset.size.short.toFixed(2) + ' ' + dataset.size.unit : 'Предустановленный' }}
          </span>
        </div>

        <div class="remove" @click.stop="$emit('remove', dataset)"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    dataset: {
      type: Object,
      default: () => {},
    },
    cardIndex: {
      type: Number,
    },
    loaded: {
      type: Boolean,
      default: false,
    },
  },
  data: () => ({
    index: 0,
  }),
  computed: {
    // getFour() {
    //   const tags = this.dataset?.tags || [];
    //   return tags.filter((item, i) => i >= this.index && i < this.index + 4);
    // },
  },
  methods: {
    click() {
      if (this.dataset.tags.length > this.index + 4) {
        this.index = this.index + 4;
      } else {
        this.index = 0;
      }
    },
  },
};
</script>

<style lang="scss">
/* datasets */
.datasets {
  height: 100%;
  padding: 0 0 0 20px;
}
.datasets > .title {
  margin-top: 20px;
}
.datasets > .inner {
  height: 100%;
  padding: 52px 0 0 0;
}
.dataset-card-container {
  height: 100%;
}
.dataset-card-wrapper {
  margin: 0 10px 0 -10px;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-start;
  align-content: stretch;
  align-items: stretch;
}
.dataset-card-item {
  padding: 0 10px 50px 10px;
  min-height: 160px;
}
.dataset-card-item.hidden {
  display: none;
}
.dataset-card {
  display: block;
  width: 160px;
  height: 130px;
  position: relative;
  border: 2px solid;
  border-width: 3px !important;
  box-sizing: border-box;
  border-radius: 4px;
  padding: 10px;
  transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
  cursor: pointer;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.dataset-card.disabled {
  opacity: 0.5;
  cursor: default;
}
.dataset-card {
  background: #242f3d;
  border-color: #6c7883;
}
.dataset-card:hover {
  border-color: #fff;
}
.dataset-card.disabled {
  border-color: #6c7883 !important;
}
.dataset-card-item.active > .dataset-card {
  border-color: #65b9f4;
}
.dataset-card-item.selected > .dataset-card {
  border-color: #3eba31;
}

.card-title {
    width: 90%;
    position:absolute;
    word-wrap: break-word;
    font-size:.875rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    word-wrap: break-word;
}
.card-body {
    height:100%;
    padding:30px 0 0 0;
    display:flex;
    flex-direction:column;
    justify-content:flex-end;
}
.card-tag {
    position:relative;
    font-size:.75rem;
    line-height:20px;
    color:#A7BED3;
    padding-left:15px;
}
.card-tag:before {
    display:block;
    content:'';
    width:8px;
    height:8px;
    margin:-4px 0 0 0;
    position:absolute;
    left:0;
    top:50%;
    background-position:center;
    background-repeat:no-repeat;
    background-size:contain;
    background-image:url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB2aWV3Qm94PSIwIDAgOCA4IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cGF0aCBkPSJNNy4wNjc1OSAwLjE1NTE0Mkw0LjQ1NDYgMC4wMDE1MzU1N0M0LjMzODIgLTAuMDA1NTg3NjcgNC4yMjE2MSAwLjAxMjEyOCA0LjExMjU3IDAuMDUzNTA3NEM0LjAwMzUzIDAuMDk0ODg2OCAzLjkwNDUzIDAuMTU4OTg0IDMuODIyMTUgMC4yNDE1NDZMMC4yNDE1NTIgMy44MjE3QzAuMDg2ODczNiAzLjk3NjY5IDAgNC4xODY3NCAwIDQuNDA1NzNDMCA0LjYyNDcyIDAuMDg2ODczNiA0LjgzNDc2IDAuMjQxNTUyIDQuOTg5NzVMMy4wMDk3MiA3Ljc1ODQxQzMuMTY0NjkgNy45MTMxMSAzLjM3NDcgOCAzLjU5MzY1IDhDMy44MTI2IDggNC4wMjI2MSA3LjkxMzExIDQuMTc3NTcgNy43NTg0MUw3Ljc1NjU3IDQuMTc4NTJDNy44MzkzIDQuMDk2MzMgNy45MDM2NiAzLjk5NzUyIDcuOTQ1MzkgMy44ODg2MkM3Ljk4NzEyIDMuNzc5NzEgOC4wMDUyOCAzLjY2MzIgNy45OTg2NyAzLjU0Njc2TDcuODQ0ODMgMC45MzE3MDlDNy44MzI4OSAwLjcyOTUxMyA3Ljc0NzE0IDAuNTM4NzM4IDcuNjAzODcgMC4zOTU1ODhDNy40NjA2IDAuMjUyNDM5IDcuMjY5NzcgMC4xNjY4NzcgNy4wNjc1OSAwLjE1NTE0MlpNNi4zNTI0OSAyLjgxNDQ2QzYuMTk1NDQgMi45NjUxOCA1Ljk4NjIyIDMuMDQ5MzUgNS43Njg1NiAzLjA0OTM1QzUuNTUwOTEgMy4wNDkzNSA1LjM0MTY4IDIuOTY1MTggNS4xODQ2NCAyLjgxNDQ2QzUuMDI5OTYgMi42NTk0NyA0Ljk0MzA5IDIuNDQ5NDIgNC45NDMwOSAyLjIzMDQzQzQuOTQzMDkgMi4wMTE0NCA1LjAyOTk2IDEuODAxNCA1LjE4NDY0IDEuNjQ2NDFDNS4zNDIwMSAxLjQ5NjIxIDUuNTUxMTggMS40MTI0MiA1Ljc2ODcgMS40MTI0MkM1Ljk4NjIyIDEuNDEyNDIgNi4xOTUzOCAxLjQ5NjIxIDYuMzUyNzUgMS42NDY0MUM2LjUwNzQgMS44MDE0MyA2LjU5NDIyIDIuMDExNSA2LjU5NDE3IDIuMjMwNDlDNi41OTQxMiAyLjQ0OTQ4IDYuNTA3MiAyLjY1OTUgNi4zNTI0OSAyLjgxNDQ2WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4=')
}
.card-extra {
    width:100%;
    padding:0 10px;
    position:absolute;
    left:0;
    bottom:-34px;
    background-color: #0e1621;
    border: 1px solid #6c788350;
    color: #A7BED3;
}
.card-extra > .wrapper {
    height:30px;
    line-height:28px;
    padding:0 10px;
    font-size:.625rem;
    font-weight:600;
    border:1px solid transparent;
    border-top:0;
    border-radius:0 0 4px 4px;
    
}
.card-extra.is-custom > .wrapper {
    margin-right:35px;
}
.card-extra.is-custom > .remove {
    width:30px;
    height:30px;
    position:absolute;
    top:0;
    right:10px;
    border-radius:0 0 4px 4px;
    transition:background-color .3s ease-in-out;
}
.card-extra.is-custom > .remove:before {
    display:block;
    content:'';
    width:16px;
    height:16px;
    margin:-8px 0 0 -8px;
    position:absolute;
    left:50%;
    top:50%;
    background-position:center;
    background-repeat:no-repeat;
    background-size:contain;
    background-image:url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTciIHZpZXdCb3g9IjAgMCAxNiAxNyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE0LjIxODUgMy4wMjIyMkgwLjk5MTk5MUMwLjcyODg5OSAzLjAyMjIyIDAuNDc2NTgyIDMuMTIxNzMgMC4yOTA1NDcgMy4yOTg4NEMwLjEwNDUxMyAzLjQ3NTk2IDAgMy43MTYxOCAwIDMuOTY2NjdDMCA0LjIxNzE1IDAuMTA0NTEzIDQuNDU3MzcgMC4yOTA1NDcgNC42MzQ0OUMwLjQ3NjU4MiA0LjgxMTYxIDAuNzI4ODk5IDQuOTExMTEgMC45OTE5OTEgNC45MTExMUgxLjMyMjY1VjEzLjc3NjNDMS4zMjQ0IDE0LjYzMDggMS42ODE3IDE1LjQ0OTggMi4zMTYzMiAxNi4wNTRDMi45NTA5NCAxNi42NTgyIDMuODExMTYgMTYuOTk4MyA0LjcwODY1IDE3SDEwLjUwMTlDMTEuMzk5NCAxNi45OTgzIDEyLjI1OTYgMTYuNjU4MiAxMi44OTQyIDE2LjA1NEMxMy41Mjg4IDE1LjQ0OTggMTMuODg2MSAxNC42MzA4IDEzLjg4NzkgMTMuNzc2M1Y0LjkxMTExSDE0LjIxODVDMTQuNDgxNiA0LjkxMTExIDE0LjczMzkgNC44MTE2MSAxNC45MiA0LjYzNDQ5QzE1LjEwNiA0LjQ1NzM3IDE1LjIxMDUgNC4yMTcxNSAxNS4yMTA1IDMuOTY2NjdDMTUuMjEwNSAzLjcxNjE4IDE1LjEwNiAzLjQ3NTk2IDE0LjkyIDMuMjk4ODRDMTQuNzMzOSAzLjEyMTczIDE0LjQ4MTYgMy4wMjIyMiAxNC4yMTg1IDMuMDIyMjJaTTExLjkwMzkgMTMuNzc2M0MxMS45MDM5IDEzLjk1MTYgMTEuODY3NiAxNC4xMjUyIDExLjc5NzIgMTQuMjg3MUMxMS43MjY3IDE0LjQ0OTEgMTEuNjIzNCAxNC41OTYyIDExLjQ5MzIgMTQuNzIwMkMxMS4zNjMxIDE0Ljg0NDEgMTEuMjA4NSAxNC45NDI0IDExLjAzODQgMTUuMDA5NUMxMC44NjgzIDE1LjA3NjYgMTAuNjg2IDE1LjExMTEgMTAuNTAxOSAxNS4xMTExSDQuNzA4NjVDNC4zMzY4MSAxNS4xMTExIDMuOTgwMjEgMTQuOTcwNSAzLjcxNzI4IDE0LjcyMDJDMy40NTQzNSAxNC40Njk4IDMuMzA2NjQgMTQuMTMwMyAzLjMwNjY0IDEzLjc3NjNWNC45MTExMUgxMS45MDM5VjEzLjc3NjNaTTMuODM1NyAwLjk0NDQ0NEMzLjgzNTcgMC42OTM5NjIgMy45NDAyMSAwLjQ1MzczOSA0LjEyNjI1IDAuMjc2NjIxQzQuMzEyMjggMC4wOTk1MDM3IDQuNTY0NiAwIDQuODI3NjkgMEgxMC4zODI4QzEwLjY0NTkgMCAxMC44OTgyIDAuMDk5NTAzNyAxMS4wODQzIDAuMjc2NjIxQzExLjI3MDMgMC40NTM3MzkgMTEuMzc0OCAwLjY5Mzk2MiAxMS4zNzQ4IDAuOTQ0NDQ0QzExLjM3NDggMS4xOTQ5MyAxMS4yNzAzIDEuNDM1MTUgMTEuMDg0MyAxLjYxMjI3QzEwLjg5ODIgMS43ODkzOSAxMC42NDU5IDEuODg4ODkgMTAuMzgyOCAxLjg4ODg5SDQuODI3NjlDNC41NjQ2IDEuODg4ODkgNC4zMTIyOCAxLjc4OTM5IDQuMTI2MjUgMS42MTIyN0MzLjk0MDIxIDEuNDM1MTUgMy44MzU3IDEuMTk0OTMgMy44MzU3IDAuOTQ0NDQ0Wk00LjY5NTQyIDEyLjUyOTZWNy40OTI1OUM0LjY5NTQyIDcuMjQyMTEgNC43OTk5NCA3LjAwMTg5IDQuOTg1OTcgNi44MjQ3N0M1LjE3MjAxIDYuNjQ3NjUgNS40MjQzMiA2LjU0ODE1IDUuNjg3NDEgNi41NDgxNUM1Ljk1MDUxIDYuNTQ4MTUgNi4yMDI4MiA2LjY0NzY1IDYuMzg4ODYgNi44MjQ3N0M2LjU3NDg5IDcuMDAxODkgNi42Nzk0MSA3LjI0MjExIDYuNjc5NDEgNy40OTI1OVYxMi41Mjk2QzYuNjc5NDEgMTIuNzgwMSA2LjU3NDg5IDEzLjAyMDMgNi4zODg4NiAxMy4xOTc1QzYuMjAyODIgMTMuMzc0NiA1Ljk1MDUxIDEzLjQ3NDEgNS42ODc0MSAxMy40NzQxQzUuNDI0MzIgMTMuNDc0MSA1LjE3MjAxIDEzLjM3NDYgNC45ODU5NyAxMy4xOTc1QzQuNzk5OTQgMTMuMDIwMyA0LjY5NTQyIDEyLjc4MDEgNC42OTU0MiAxMi41Mjk2Wk04LjUzMTEyIDEyLjUyOTZWNy40OTI1OUM4LjUzMTEyIDcuMjQyMTEgOC42MzU2MyA3LjAwMTg5IDguODIxNjcgNi44MjQ3N0M5LjAwNzcgNi42NDc2NSA5LjI2MDAyIDYuNTQ4MTUgOS41MjMxMSA2LjU0ODE1QzkuNzg2MiA2LjU0ODE1IDEwLjAzODUgNi42NDc2NSAxMC4yMjQ2IDYuODI0NzdDMTAuNDEwNiA3LjAwMTg5IDEwLjUxNTEgNy4yNDIxMSAxMC41MTUxIDcuNDkyNTlWMTIuNTI5NkMxMC41MTUxIDEyLjc4MDEgMTAuNDEwNiAxMy4wMjAzIDEwLjIyNDYgMTMuMTk3NUMxMC4wMzg1IDEzLjM3NDYgOS43ODYyIDEzLjQ3NDEgOS41MjMxMSAxMy40NzQxQzkuMjYwMDIgMTMuNDc0MSA5LjAwNzcgMTMuMzc0NiA4LjgyMTY3IDEzLjE5NzVDOC42MzU2MyAxMy4wMjAzIDguNTMxMTIgMTIuNzgwMSA4LjUzMTEyIDEyLjUyOTZaIiBmaWxsPSIjNjVCOUY0Ii8+Cjwvc3ZnPg==')
}
</style>