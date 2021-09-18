<template>
  <div class="block-handlers">
    <div class="block-handlers__header">
      <p>Обработчики</p>
      <Fab @click="handleAdd" />
    </div>
    <scrollbar :ops="ops">
      <div class="block-handlers__content">
        <template v-for="(handler, index) in handlers">
          <CardHandler
            v-bind="handler"
            :key="'handler' + index"
            @click-btn="handleClick($event, handler.id)"
            @click.native="select(handler.id)"
          >
            <template v-slot:header>{{ `${handler.name} ${handler.id}` }}</template>

          </CardHandler>
        </template>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Fab from '../components/forms/Fab';
import CardHandler from '../components/card/CardHandler';

export default {
  name: 'block-handlers',
  components: {
    Fab,
    CardHandler,
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: true,
        scrollingY: false,
      },
    },
    colors: [
      '#1ea61d',
      '#a51da6',
      '#0d6dea',
      '#fecd05',
      '#d72239',
      '#054f1d',
      '#630e76',
      '#031e70',
      '#b78b01',
      '#660634',
      '#86e372',
      '#e473d0',
      '#6bb5f9',
      '#ffe669',
      '#f38079',
    ],
    table: {},
  }),
  computed: {
    handlers: {
      set(value) {
        this.$store.dispatch('tables/setHandlers', value);
      },
      get() {
        return this.$store.getters['tables/getHandlers'];
      },
    },
  },
  created() {
    const files = this.$store.getters['datasets/getFilesSource'];
    console.log(files);
    this.table = files
      .filter(item => item.type === 'table')
      .reduce((obj, item) => {
        obj[item.title] = [];
        return obj;
      }, {});
  },
  methods: {
    select(id) {
      this.handlers = this.handlers.map(item => {
        item.active = item.id === id;
        return item;
      });
    },
    deselect() {
      this.handlers = this.handlers.map(item => {
        item.active = false;
        return item;
      });
    },
    handleAdd() {
      console.log(this.table);
      this.deselect();
      this.handlers.push({
        id: this.handlers.length + 1,
        name: 'Name',
        active: true,
        color: this.colors[this.handlers.length],
        layer: 'String',
        type: 'String',
        table: JSON.parse(JSON.stringify(this.table)),
      });
      console.log(this.handlers);
    },
    handleClick(e, id) {
      if (e === 'remove') {
        this.deselect();
        this.handlers = this.handlers.filter(item => item.id !== id);
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.block-handlers {
  margin: 10px auto;
  height: 400px;
  padding: 0 0 25px;
  p {
    font-size: 14px;
  }
  &__header {
    height: 32px;
    background: #242f3d;
    display: flex;
    justify-content: center;
    gap: 10px;
    align-items: center;
  }
  &__content {
    margin: 10px auto;
    display: flex;
    justify-content: center;
    height: 350px;
  }
}
</style>