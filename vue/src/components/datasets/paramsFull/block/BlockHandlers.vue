<template>
  <div :class="['block-handlers', { 'block-handlers--hide': !show }]">
    <div class="block-handlers__header">
      <div class="block-handlers__item">
        <Fab @click="handleAdd" />
        <p>Обработчики</p>
        <div class="block-handlers__item--left" @click="show = !show">
          <i :class="['t-icon icon-collapsable', { rotate: show }]"></i>
        </div>
      </div>
    </div>
    <scrollbar v-if="show" :ops="ops">
      <div class="block-handlers__content">
        <template v-for="(handler, index) in handlers">
          <CardHandler
            v-bind="handler"
            :key="'handler' + index"
            @click-btn="handleClick($event, handler.id)"
            @click.native="select(handler.id)"
          >
            <template v-slot:header>{{ `${handler.name}` }}</template>
            <template v-slot:default="{ data: { parameters, errors } }">
              <template v-for="(data, index) of formsHandler">
                <t-auto-field-handler
                  v-bind="data"
                  :parameters="parameters"
                  :errors="errors"
                  :key="handler.color + index"
                  :idKey="'key_' + index"
                  :id="handler.id"
                  root
                  @change="change"
                />
              </template>
            </template>
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
    show: true,
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
    formsHandler() {
      return this.$store.getters['datasets/getFormsHandler'];
    },
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
    change({ id, value, name }) {
      const index = this.handlers.findIndex(item => item.id === id);
      if (name === 'name') {
        this.handlers[index].name = value;
      }
      if (name === 'type') {
        this.handlers[index].type = value;
      }
      if (this.handlers[index]) {
        this.handlers[index].parameters[name] = value;
      }
      this.handlers = [...this.handlers];
    },
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
      if (!this.show) return;
      console.log(this.table);
      this.deselect();
      let maxID = Math.max(0, ...this.handlers.map(o => o.id));
      this.handlers.push({
        id: maxID + 1,
        name: 'Name_' + (maxID + 1),
        active: true,
        color: this.colors[this.handlers.length],
        layer: (this.handlers.length + 1).toString(),
        type: '',
        table: JSON.parse(JSON.stringify(this.table)),
        parameters: {},
      });
      console.log(this.handlers);
    },
    handleClick(event, id) {
      if (event === 'remove') {
        this.deselect();
        this.handlers = this.handlers.filter(item => item.id !== id);
      }
      console.log(event);
      if (event === 'copy') {
        this.deselect();
        const copy = JSON.parse(JSON.stringify(this.handlers.filter(item => item.id == id)));
        let maxID = Math.max(0, ...this.handlers.map(o => o.id));
        copy[0].id = maxID + 1
        copy[0].name = 'Name_' + (maxID + 1),
        this.handlers = [...this.handlers, ...copy];
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
  &--hide {
    height: 30px;
  }
  .rotate {
    transform: rotate(180deg);
  }
  &__header {
    user-select: none;
    height: 32px;
    background: #242f3d;
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
    // gap: 10px;
    align-items: center;
    padding: 0 10px;
  }
  &__item {
    width: 50%;
    display: flex;
    align-items: center;
    padding: 0 7px;
    p {
      margin-left: 10px;
      font-style: normal;
      font-weight: normal;
      font-size: 12px;
      line-height: 16px;
    }
    &--left {
      height: 30px;
      display: flex;
      align-items: center;
      margin-left: auto;
      i {
        height: 12px;
      }
    }
  }
  &__content {
    margin: 10px auto;
    display: flex;
    justify-content: center;
    height: 350px;
  }
}
</style>