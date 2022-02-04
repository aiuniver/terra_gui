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
}
</style>