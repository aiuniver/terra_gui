<template>
  <div class="block-handlers">
      <div class="block-handlers__header">
          <p>Обработчики</p>
          <Fab @click="handlers.push(handlers.length)" />
      </div>
      <scrollbar :ops="ops">
        <div class="block-handlers__content">
            <CardLayer v-for="(val, idx) in handlers" :key="idx" @click-btn="handleClick($event, idx)">
                <template v-slot:header>Обработчик {{idx+1}}</template>
                <t-input :inline="true"/>
            </CardLayer>
        </div>
      </scrollbar>
  </div>
</template>

<script>
import Fab from '../components/forms/Fab.vue'
import CardLayer from '../components/card/CardLayer.vue'

export default {
    name: 'block-handlers',
    components: {
        Fab,
        CardLayer
    },
    data: () => ({
        handlers: [],
        ops: {
            scrollPanel: {
                scrollingX: true,
                scrollingY: false,
            },
        }
    }),
    methods: {
        handleClick(e, idx) {
            if (e === 'remove') return this.handlers.splice(idx, 1)
            this.handlers.splice(idx, 0, this.handlers[idx])
        }
    }
}
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