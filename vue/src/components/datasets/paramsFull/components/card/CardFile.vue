<template>
  <div class="card-file" :style="bc">
    <div v-if="active" class="card-file__header" :style="bg">{{ 'Входные данные ' + sloy }}</div>
    <div :class="['card-file__body', type]"></div>
    <div class="card-file__footer">{{ label }}</div>
  </div>
</template>

<script>
export default {
  name: 'card-file',
  props: {
    color: {
      type: String,
      default: '',
    },
    active: Boolean,
    label: String,
    name: String,
    type: String,
    sloy: Number,
  },
  computed: {
    bg() {
      return { backgroundColor: this.active ? this.color : '' };
    },
    bc() {
      return { borderColor: this.active ? this.color : '' };
    },
  },
};
</script>

<style lang="scss" scoped>
.card-file {
  flex: 0 0 152px;
  position: relative;
  width: 152px;
  height: 152px;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  background-color: #17212b;
  margin: 0 5px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  &__footer,
  &__header {
    position: relative;
    height: 24px;
    background-color: #242f3d;
    width: 100%;
    border-radius: 0 0 3px 3px;
    font-family: Open Sans;
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 16px;
  }
  &__header {
    top: 0;
    border-radius: 3px 3px 0 0;
    padding: 2px 6px 4px 6px;
  }
  &__body {
    height: 100%;
    width: 100%;
    display: inline-block;
    background-repeat: no-repeat;
    background-position: 50% 40%;
    background-size: 39px 39px;
  }
  &__footer {
    bottom: 0;
    border-radius: 0 0 3px 3px;
    padding: 4px 6px 2px 6px;
  }
}
.audio {
  background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjYiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCAyNiA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyLjk5OTkgNC44MzMzM1YyMy4zNTgzQzEwLjk2MzMgMjIuMTg4MyA4LjQ0OTkzIDIxLjczMzMgNS43ODQ5MyAyMi42NjVDMi44ODE2IDIzLjcwNSAwLjY0OTkzNSAyNi4yODMzIDAuMTI5OTM1IDI5LjMxNjdDLTAuODY2NzMyIDM1LjI1MzMgNC4xNTk5MyA0MC4zMjMzIDEwLjA3NDkgMzkuMzkxN0MxNC4zMjE2IDM4LjcyIDE3LjMzMzMgMzQuODIgMTcuMzMzMyAzMC41MDgzVjkuMTY2NjdIMjEuNjY2NkMyNC4wNDk5IDkuMTY2NjcgMjUuOTk5OSA3LjIxNjY3IDI1Ljk5OTkgNC44MzMzM0MyNS45OTk5IDIuNDUgMjQuMDQ5OSAwLjUgMjEuNjY2NiAwLjVIMTcuMzMzM0MxNC45NDk5IDAuNSAxMi45OTk5IDIuNDUgMTIuOTk5OSA0LjgzMzMzWiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K);
}
.text {
  background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTI0LjcwMTcgMC41SDQuODMzMzNDMi40NSAwLjUgMC41IDIuNDUgMC41IDQuODMzMzNWMzUuMTY2N0MwLjUgMzcuNTUgMi40NSAzOS41IDQuODMzMzMgMzkuNUgzNS4xNjY3QzM3LjU1IDM5LjUgMzkuNSAzNy41NSAzOS41IDM1LjE2NjdWMTUuMjk4M0MzOS41IDE0LjE1IDM5LjA0NSAxMy4wNDUgMzguMjIxNyAxMi4yNDMzTDI3Ljc1NjcgMS43NzgzM0MyNi45NTUgMC45NTUgMjUuODUgMC41IDI0LjcwMTcgMC41Wk0xMS4zMzMzIDI2LjVIMjguNjY2N0MyOS44NTgzIDI2LjUgMzAuODMzMyAyNy40NzUgMzAuODMzMyAyOC42NjY3QzMwLjgzMzMgMjkuODU4MyAyOS44NTgzIDMwLjgzMzMgMjguNjY2NyAzMC44MzMzSDExLjMzMzNDMTAuMTQxNyAzMC44MzMzIDkuMTY2NjcgMjkuODU4MyA5LjE2NjY3IDI4LjY2NjdDOS4xNjY2NyAyNy40NzUgMTAuMTQxNyAyNi41IDExLjMzMzMgMjYuNVpNMTEuMzMzMyAxNy44MzMzSDI4LjY2NjdDMjkuODU4MyAxNy44MzMzIDMwLjgzMzMgMTguODA4MyAzMC44MzMzIDIwQzMwLjgzMzMgMjEuMTkxNyAyOS44NTgzIDIyLjE2NjcgMjguNjY2NyAyMi4xNjY3SDExLjMzMzNDMTAuMTQxNyAyMi4xNjY3IDkuMTY2NjcgMjEuMTkxNyA5LjE2NjY3IDIwQzkuMTY2NjcgMTguODA4MyAxMC4xNDE3IDE3LjgzMzMgMTEuMzMzMyAxNy44MzMzWk0xMS4zMzMzIDkuMTY2NjdIMjIuMTY2N0MyMy4zNTgzIDkuMTY2NjcgMjQuMzMzMyAxMC4xNDE3IDI0LjMzMzMgMTEuMzMzM0MyNC4zMzMzIDEyLjUyNSAyMy4zNTgzIDEzLjUgMjIuMTY2NyAxMy41SDExLjMzMzNDMTAuMTQxNyAxMy41IDkuMTY2NjcgMTIuNTI1IDkuMTY2NjcgMTEuMzMzM0M5LjE2NjY3IDEwLjE0MTcgMTAuMTQxNyA5LjE2NjY3IDExLjMzMzMgOS4xNjY2N1oiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==);
}
.image {
  background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTM5LjUgMzUuMTY2N1Y0LjgzMzMzQzM5LjUgMi40NSAzNy41NSAwLjUgMzUuMTY2NyAwLjVINC44MzMzM0MyLjQ1IDAuNSAwLjUgMi40NSAwLjUgNC44MzMzM1YzNS4xNjY3QzAuNSAzNy41NSAyLjQ1IDM5LjUgNC44MzMzMyAzOS41SDM1LjE2NjdDMzcuNTUgMzkuNSAzOS41IDM3LjU1IDM5LjUgMzUuMTY2N1pNMTMuMjgzMyAyNC4yOUwxNy44MzMzIDI5Ljc3MTdMMjQuNTUgMjEuMTI2N0MyNC45ODMzIDIwLjU2MzMgMjUuODUgMjAuNTYzMyAyNi4yODMzIDIxLjE0ODNMMzMuODg4MyAzMS4yODgzQzM0LjQzIDMyLjAwMzMgMzMuOTEgMzMuMDIxNyAzMy4wMjE3IDMzLjAyMTdINy4wNDMzM0M2LjEzMzMzIDMzLjAyMTcgNS42MzUgMzEuOTgxNyA2LjE5ODMzIDMxLjI2NjdMMTEuNTkzMyAyNC4zMzMzQzEyLjAwNSAyMy43NyAxMi44MjgzIDIzLjc0ODMgMTMuMjgzMyAyNC4yOVoiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==);
}
.folder {
  background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAyMCAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE4IDJIMTBMOC41OSAwLjU5QzguMjEgMC4yMSA3LjcgMCA3LjE3IDBIMkMwLjkgMCAwLjAwOTk5OTk5IDAuOSAwLjAwOTk5OTk5IDJMMCAxNEMwIDE1LjEgMC45IDE2IDIgMTZIMThDMTkuMSAxNiAyMCAxNS4xIDIwIDE0VjRDMjAgMi45IDE5LjEgMiAxOCAyWk0xNyAxNEgzQzIuNDUgMTQgMiAxMy41NSAyIDEzVjVDMiA0LjQ1IDIuNDUgNCAzIDRIMTdDMTcuNTUgNCAxOCA0LjQ1IDE4IDVWMTNDMTggMTMuNTUgMTcuNTUgMTQgMTcgMTRaIiBmaWxsPSIjQTdCRUQzIi8+Cjwvc3ZnPgo=);
}
</style>