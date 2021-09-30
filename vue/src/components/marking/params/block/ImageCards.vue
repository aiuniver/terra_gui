<template>
    <div class="t-images">
        <div class="t-images__header">
            <button>Предыдущий</button>
            <p>Классификация треков</p>
            <button>Следующий</button>
        </div>
        <t-range :minMax="[0, 812]" :showValue="true" class="t-images__range"/>
        <scrollbar>
            <div class="t-images__cards">
                <t-card-images v-for="val in blockNum"
                :key="val"
                />
            </div>
        </scrollbar>
        <div class="t-images__footer">
            <p>Показывать:</p>
            <div class="t-images__footer--btns">
                <button v-for="val in btns"
                :key="val"
                :class="{ selected: val === blockNum }"
                @click="blockNum = val"
                >{{ val }}</button>
            </div>
        </div>
    </div>
</template>

<script>
import tCardImages from '@/components/marking/params/components/card/CardImages.vue';
import tRange from '../components/Range.vue'

export default {
    name: 'image-cards',
    components: {
        tCardImages,
        tRange
    },
    data: () => ({
        btns: [1, 2, 4, 6],
        blockNum: 4
    })
}
</script>

<style lang="scss" scoped>
.t-images {
    padding: 20px;
    height: 100%;
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 20px;
    &__cards {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 30px;
        height: 100%;
        padding: 10px 0;
    }
    &__header {
        display: flex;
        justify-content: space-between;
        p, button {
            font-size: 14px;
            line-height: 22px;
        }
        button {
            flex: 0 0 130px;
            background: none;
            height: 24px;
        }
    }
    &__range {
        max-width: 660px;
        margin: auto;
    }
    &__footer {
        width: 100%;
        text-align: center;
        &--btns {
            margin-top: 10px;
            display: flex;
            justify-content: center;
            gap: 10px;
            button {
                flex: 0 0 24px;
                width: 24px;
                height: 24px;
                font-size: 14px;
                line-height: 100%;
                background: none;
                vertical-align: middle;
                &.selected {
                    background: #65B9F4;
                }
            }   
        }
    }
}
</style>